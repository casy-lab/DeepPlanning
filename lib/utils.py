import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.losses import Loss

try:
    from pose import Pose
except:
    from .pose import Pose

class TrajectoryCostLoss(Loss):
    def __init__(self, params, ref_frame='bf', state_dim=3):
        super(TrajectoryCostLoss, self).__init__()
        self.params = params
        self.ref_frame = ref_frame
        self.state_dim = state_dim
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def add_pointclouds(self, pc):
        self.pointclouds = pc

    def compute_traj_cost(self, exp_id, state, pred_trajs):
        num_modes = pred_trajs.shape[0]
        costs = []
        for k in range(num_modes):
            traj_cost = tf.py_function(func=self.compute_single_traj_cost, inp=[exp_id, state, pred_trajs[k]], Tout=tf.float32)
            traj_cost.set_shape((1,))
            costs.append(traj_cost)

        return tf.stack(costs)

    def compute_single_traj_cost(self, exp_id, state, traj):
        pcd_idx = int(exp_id.numpy()[0])  # Matlab Indexes
        pcd_tree = self.pointclouds[pcd_idx-1]

        traj = traj.numpy()
        state = state.numpy()
        traj_len = traj.shape[0] // self.state_dim

        collision_threshold = 0.8
        quadrotor_size = 0.25

        # Convert Traj. to World Frame
        traj = np.reshape(traj, ((-1, traj_len)))
        traj = to_world_frame(traj, sp=state[:3].reshape((3, 1)), sa=state[3:].reshape((3, 3)), frame=self.ref_frame)

        cost = 0
        for j in range(traj_len):
            [_, __, dists_squared] = pcd_tree.search_radius_vector_3d(traj[:, j], collision_threshold)

            if len(dists_squared) > 0:
                dist = np.sqrt(np.min(dists_squared))
                if dist < quadrotor_size:
                    # This Point is in Contact With The Quadrotor
                    # Parabolic Cost With Vertex in 4
                    cost += ((-dist**2/(quadrotor_size**2)) + 4)
                else:
                    # Linear Decrease With 1 at collision_threshold
                    cost += (((quadrotor_size - dist)/(collision_threshold - quadrotor_size)) + 2)

        # Average Cost
        cost = cost/traj_len
        return np.array(cost, dtype=np.float32).reshape((1,))

    def call(self, ids_and_states, y_pred):
        ids = ids_and_states[0]
        states = ids_and_states[1]
        alphas = y_pred[:, :, 0]

        ids = tf.reshape(ids, (-1, 1))
        batch_size = ids.shape[0]
        traj_costs = []
        for k in range(batch_size):
            traj_costs.append(tf.stop_gradient(self.compute_traj_cost(ids[k], states[k], y_pred[k, :, 1:])))

        traj_costs = tf.stack(traj_costs)
        traj_cost_loss = self.params.lambda_2*self.mse_loss(traj_costs, alphas)
        return traj_cost_loss

class MixtureSpaceLoss(Loss):
    def __init__(self, params, modes=3):
        super(MixtureSpaceLoss, self).__init__()
        self.params = params
        self.space_loss = DiscretePositionLoss(self.params)
        self.modes = modes
        self.margin = 2

    def call(self, y_true, y_pred):
        mode_losses = []
        alphas = []

        for j in range(self.modes):
            pred_len = y_pred.shape[-1]
            alpha = tf.reshape(y_pred[:, j, 0], (-1, 1))
            pred = tf.reshape(y_pred[:, j, 1:], (-1, pred_len - 1))
            mode_loss = []

            for k in range(y_true.shape[1]):  # Number of Trajs
                mode_loss.append(self.space_loss(y_true[:, k], pred))

            mode_loss = tf.concat(mode_loss, axis=1)  # [B,K]
            mode_loss = tf.expand_dims(mode_loss, axis=-1)

            alphas.append(alpha)
            mode_losses.append(mode_loss)

        alphas = tf.concat(alphas, axis=-1)  # [B,M]
        mode_losses = tf.concat(mode_losses, axis=-1)  # [B,K,M]
        max_idx = tf.argmin(mode_losses, axis=-1)  # [B,K]
        loss_matrix = tf.zeros_like(alphas)

        for k in range(y_true.shape[1]):
            selection_matrix = tf.one_hot(max_idx[:, k], depth=self.modes)
            loss_matrix = loss_matrix + (selection_matrix*mode_losses[:, k, :]*(1 - self.params.epsilon))

        # Considering All Selected Modes Over All Possible GT Trajectories
        final_selection_matrix = tf.cast(tf.greater(loss_matrix, 0.0), tf.float32)  # [B,M]

        # Give a Cost to All Trajectories Which Received NO Vote
        if self.modes > 1:
            relaxed_cost_matrix = (1.0 - final_selection_matrix)*mode_losses[:, 0, :]*self.params.epsilon/(float(self.modes) - 1.0)
            final_cost_matrix = loss_matrix + relaxed_cost_matrix
        else:
            final_cost_matrix = loss_matrix

        return tf.reduce_mean(tf.reduce_mean(final_cost_matrix, axis=-1))

class DiscretePositionLoss(Loss):
    def __init__(self, params):
        super(DiscretePositionLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
        self.params = params
        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # Compute Normalization Factor (Square Lenght of the GT Traj)
        average_loss = self.mse_loss(y_true, y_pred)
        average_loss = tf.expand_dims(average_loss, axis=-1)

        # Compute Batch Average Loss
        final_loss = average_loss*self.params.lambda_1
        return final_loss

def to_world_frame(t, sp, sa, frame='bf'):
    if frame == 'bf':
        T_W_S = Pose(sa, sp)
    else:
        assert False, "Unknown reference frame."

    for i in range(t.shape[1]):
        bf_pose = Pose(np.eye(3), t[:, i].reshape((3, 1)))
        wf_pose = T_W_S*bf_pose
        t[:, i] = np.squeeze(wf_pose.t)

    return t

def convert_to_npy(filename, num_states=10):
    npy_file = filename[:-4] + ".npy"
    if os.path.isfile(npy_file):
        return npy_file

    df = pd.read_csv(filename, delimiter=',')
    if df.shape[0] == 0:
        return "None"

    x_pos_name = "pos_x_{}"
    y_pos_name = "pos_y_{}"
    z_pos_name = "pos_z_{}"

    x_pos_load = []
    y_pos_load = []
    z_pos_load = []
    for n in np.arange(1, num_states + 1):
        x_pos_load.append(x_pos_name.format(n))
        y_pos_load.append(y_pos_name.format(n))
        z_pos_load.append(z_pos_name.format(n))

    # Load Data From CSV
    rel_cost = df["rel_cost"].values
    x_pos = df[x_pos_load].values
    y_pos = df[y_pos_load].values
    z_pos = df[z_pos_load].values

    # Each Row is a Different Trajectory
    full_trajectory = np.column_stack((x_pos, y_pos, z_pos, rel_cost))

    # Write npy
    assert full_trajectory.shape[-1] == 3*num_states + 1

    np.save(npy_file, full_trajectory)
    return npy_file