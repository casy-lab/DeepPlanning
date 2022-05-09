import os
import cv2
import fnmatch

import numpy as np
import pandas as pd
import tensorflow as tf
import open3d as o3d
from scipy.spatial.transform import Rotation as R

try:
    from .utils import convert_to_npy
except:
    from utils import convert_to_npy

class Dataset:
    def __init__(self, params, dir, is_training=True):
        self.params = params
        self.dir = dir
        self.is_training = is_training
        self.samples = 0
        self.experiments = []
        self.img_filenames = []
        self.depth_filenames = []
        self.imu_obs = []
        self.traj_fnames = []
        self.traj_idx_num = []
        self.pointclouds = []

        for root, dirs, files in os.walk(dir, topdown=True, followlinks=True):
            for d in dirs:
                if d.startswith(self.params.img_rootname):
                    exp_dir = os.path.join(root, d)
                    self.experiments.append(os.path.abspath(exp_dir))

        self.rollout_id = 0
        self.num_experiments = len(self.experiments)
        for exp_dir in self.experiments:
            self.decode_experiment(exp_dir)
        if self.samples == 0:
            raise IOError("Did not find any file in the dataset folder")
        
        print('Found {} images  belonging to {} experiments:'.format(self.samples, self.num_experiments))
        self.build_dataset()

    def build_dataset(self):
        # Need to Take Care That rollout_idxs are Consistent
        self.imu_obs = np.stack(self.imu_obs)

        # From Here on, self.imu_obs Has Shape [n_samples, 22]
        self.imu_obs = self.imu_obs.astype(np.float32)

        # Preprocess Dataset to Create Sampling List
        self.preprocess_dataset()

        # Form Training Batches
        dataset = tf.data.Dataset.range(self.samples)

        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.samples)

        dataset = dataset.map(self.dataset_map, num_parallel_calls=4)
        dataset = dataset.batch(self.params.batch_size, drop_remainder=not self.is_training)
        dataset = dataset.prefetch(buffer_size=4)
        self.batched_dataset = dataset

    def decode_experiment(self, dir_subpath):
        all_images = sorted(os.listdir(os.path.join(dir_subpath, 'img')))
        all_images = [f for f in all_images if ("frame_left" in f)]

        all_traj = os.listdir(os.path.join(dir_subpath, 'trajectories'))
        all_traj = [f for f in all_traj if f.startswith('trajectories_{}'.format(self.params.ref_frame)) and f.endswith('.{}'.format(self.params.data_format))]

        # Check and Convert if Needed
        all_traj_np = []
        for filename in all_traj:
            all_traj_np.append(convert_to_npy(os.path.join(dir_subpath, 'trajectories', filename), num_states=self.params.out_seq_len))

        # Get Odometry Data
        data_name = os.path.join(dir_subpath, "odometry.csv")
        assert os.path.isfile(data_name), "Not Found Odometry Data File"

        df = pd.read_csv(data_name, delimiter=',')
        df = self.quat2rot(df)
        num_files = df.shape[0]

        features_odom = ["pos_x", "pos_y", "pos_z",
                         "r_0", "r_1", "r_2", "r_3", "r_4", "r_5", "r_6", "r_7", "r_8",
                         "vel_x", "vel_y", "vel_z"]
        features_odom_v = df[features_odom].values

        if self.params.velocity_frame == 'bf':
            R_W_C = features_odom_v[:,3:12].reshape((-1,3,3))

            # Transform in Rotation
            for k in range(R_W_C.shape[0]):
                R_W_C_k = R_W_C[k]
                R_C_W_k = R_W_C_k.T

                v_k_w = features_odom_v[k,12:15].reshape((3,1))
                v_k_c = R_C_W_k@v_k_w

                # Apply Transform in Place
                features_odom_v[k,12:15] = v_k_c.T

        elif self.params.velocity_frame != 'wf':
            raise IOError("Velocity Frame is Unclear")

        num_images = len(all_images)
        assert num_files == num_images, "Number of Images Does Not Match With Odometry Measures"

        # Get the Trajectory
        ref_fname = os.path.join(dir_subpath, "reference_trajectory.csv")
        assert os.path.isfile(ref_fname), "Not Reference Trajectory File"

        ref_df = pd.read_csv(ref_fname, delimiter=',')
        vel_names = ['pos_x', 'pos_y', 'pos_z']
        reference_positions = ref_df[vel_names].values

        features_pos = ["pos_x", "pos_y", "pos_z"]
        drone_pos = df[features_pos].values
        reference_progress, reference_direction = self.calculate_ref_dir(drone_pos=drone_pos, ref_pos=reference_positions)

        # Get PointCloud
        pc_name = os.path.join(dir_subpath, "pointcloud-unity.ply")
        assert os.path.isfile(pc_name), "Not Found PointCloud file"

        # Ready to Append Data
        self.rollout_id += 1
        pc = o3d.io.read_point_cloud(pc_name)
        pc_tree = o3d.geometry.KDTreeFlann(pc)
        self.pointclouds.append(pc_tree)

        for k in range(num_files):
            is_valid = False
            img_fname = os.path.join(dir_subpath, "img", "frame_left_{:08d}.{}".format(k+1, self.params.img_format))
            traj_fname = os.path.join(dir_subpath, "trajectories", "trajectories_{}_{:08d}.{}".format(self.params.ref_frame, k, "npy"))
            depth_fname = os.path.join(dir_subpath, "img", "depth_{:08d}.{}".format(k+1, "tif"))

            if os.path.isfile(img_fname) and os.path.isfile(traj_fname) and os.path.isfile(depth_fname) and reference_progress[k] > 50:
                is_valid = True

            if is_valid:
                self.traj_fnames.append(traj_fname)
                self.traj_idx_num.append(k)

                if self.params.use_rgb:
                    self.img_filenames.append(img_fname)
                if self.params.use_depth:
                    self.depth_filenames.append(depth_fname)

                goal_dir = self.adapt_reference_frame(features_odom_v[k][3:12], reference_direction[k])
                self.imu_obs.append(np.concatenate((features_odom_v[k], goal_dir, [self.rollout_id])))
                self.samples += 1

    def quat2rot(self, df):
        rot_lod = ["q_x", "q_y", "q_z", "q_w"]
        rot_cvt_fmt = "r_{}"
        rot_val = df[rot_lod].values.tolist()

        quat_mat = list(map(lambda x: R.from_quat(x).as_matrix().reshape((9,)).tolist(), rot_val))
        quat_mat = np.array(quat_mat, dtype=np.float32)

        # Load Into the Panda Dataset
        for j in range(9):
            df[rot_cvt_fmt.format(j)] = quat_mat[:, j]

        return df

    def calculate_ref_dir(self, drone_pos, ref_pos):
        reference_progress = [1]
        goal_dir = ref_pos[np.minimum(int(50*self.params.future_time),ref_pos.shape[0]-1)] - drone_pos[0]
        goal_dir = goal_dir / np.linalg.norm(goal_dir)
        reference_direction = [goal_dir]

        for j in range(1, drone_pos.shape[0]):
            quad_position = drone_pos[j]
            current_idx = reference_progress[-1]
            reference_position = ref_pos[current_idx]
            distance = np.linalg.norm(reference_position - quad_position)

            if current_idx + 1 >= ref_pos.shape[0]:
                reference_progress.append(current_idx)
                goal_dir = ref_pos[current_idx] - quad_position
                goal_dir = goal_dir / np.linalg.norm(goal_dir)
                reference_direction.append(goal_dir)
            else:
                for k in range(current_idx + 1, ref_pos.shape[0]):
                    reference_position = ref_pos[k]
                    next_point_distance = np.linalg.norm(reference_position - quad_position)
                    if next_point_distance > distance:
                        reference_progress.append(k-1)
                        future_idx = np.minimum(k-1 + int(50*self.params.future_time), ref_pos.shape[0] -1)
                        goal_dir = ref_pos[future_idx] - quad_position
                        goal_dir = goal_dir / np.linalg.norm(goal_dir)
                        reference_direction.append(goal_dir)
                        break
                    else:
                        distance = next_point_distance
                        if k == ref_pos.shape[0] -1:
                            # Distance of All Next Points is Larger than current_Idx
                            reference_progress.append(current_idx)
                            goal_dir = ref_pos[current_idx] - quad_position
                            goal_dir = goal_dir / np.linalg.norm(goal_dir)
                            reference_direction.append(goal_dir)

        assert len(reference_progress) == len(reference_direction)
        assert len(reference_progress) == drone_pos.shape[0]
        return reference_progress, reference_direction

    def adapt_reference_frame(self, rotation_matrix, reference_velocity):
        if self.params.velocity_frame == 'wf':
            return reference_velocity
        elif self.params.velocity_frame == 'bf':
            R_W_C = rotation_matrix.reshape((3, 3))
            v_C = R_W_C.T@reference_velocity.reshape([3,1])
            return np.squeeze(v_C)
        else:
            raise IOError("Reference Frame not Recognized")

    def preprocess_dataset(self):
        # Create a Stacked Sequence Considering the rollout_idx
        input_frequency = self.params.data_save_freq
        self.processed_idxs = []
        for k in range(self.imu_obs.shape[0]):
            if k % 3000 == 0:
                print("Built {:.2f}% of the dataset".format(k / (self.imu_obs.shape[0]) * 100), end='\r')

            current_rollout_idx = self.imu_obs[k, -1]
            idx = 0
            idx_seq = []
            while len(idx_seq) < self.params.seq_len:
                if k - idx < 0:
                    # Initial Transient, Append the Same Up to Fill
                    idx_seq.append(idx_seq[-1])
                    continue

                current_idx = k - idx
                rollout_idx = self.imu_obs[current_idx, -1]  # Last Reports Experiment Number
                if rollout_idx == current_rollout_idx:
                    # Same experiment
                    idx_seq.append(current_idx)
                else:
                    # We Are in a Transient, Duplicate Last Element
                    idx_seq.append(idx_seq[-1])

                idx += int(input_frequency/self.params.seq_len) # Sampling at self.params.seq_len Frequency

            assert len(idx_seq) == self.params.seq_len, "Something Went Wrong in Frame Processing"
            self.processed_idxs.append(idx_seq)

        self.processed_idxs = np.stack(self.processed_idxs).astype(np.int32)
        assert self.processed_idxs.shape[0] == self.imu_obs.shape[0]

    def dataset_map(self, sample_num):
        gt_traj = tf.py_function(func=self.load_trajectory, inp=[sample_num], Tout=tf.float32)

        # traj_idx_num (passed only for visualization)
        traj_num = tf.gather(self.traj_idx_num, sample_num)

        imu_seq = []
        img_seq = []
        depth_seq = []
        idx_seq = tf.gather(self.processed_idxs, sample_num)
        rollout_id = tf.gather(self.imu_obs, idx_seq[0])[-1]

        for i in reversed(range(self.params.seq_len)):
            # Last is rollout_id, exclude
            imu_o = tf.gather(self.imu_obs, idx_seq[i])[:-1]
            imu_seq.append(imu_o)
            if self.params.use_rgb:
                img = tf.py_function(func=self.decode_img_cv2, inp=[idx_seq[i]], Tout=tf.float32)
                img_seq.append(img)
            if self.params.use_depth:
                depth = tf.py_function(func=self.decode_depth_cv2, inp=[idx_seq[i]], Tout=tf.float32)
                depth_seq.append(depth)

        # Concat and Play
        imu_seq = tf.stack(imu_seq)  # [seq_len, N]
        if self.params.use_rgb and self.params.use_depth:
            img_seq = tf.stack(img_seq)  # [seq_len, img_height, img_width, 3]
            depth_seq = tf.stack(depth_seq)  # [seq_len, img_height, img_width, 3]
            frame_seq = (img_seq, depth_seq)
        elif self.params.use_rgb and (not self.params.use_depth):
            frame_seq = tf.stack(img_seq)
        elif self.params.use_depth and (not self.params.use_rgb):
            frame_seq = tf.stack(depth_seq)

        return (imu_seq, frame_seq, rollout_id), gt_traj, traj_num

    def load_trajectory(self, sample_num):
        sample_num_np = sample_num.numpy()
        fname = self.traj_fnames[sample_num_np]
        all_traj = np.load(fname)
        k = np.minimum(self.params.top_trajectories, all_traj.shape[0])
        traj_set = np.zeros((self.params.top_trajectories, 3*self.params.out_seq_len))

        traj_set[:k] = all_traj[:k,:-1]
        if k < self.params.top_trajectories:
            # Copy Some
            traj_set[k:] = np.repeat(np.expand_dims(traj_set[0], 0), [self.params.top_trajectories - k], axis=0)
        return np.array(traj_set, dtype=np.float32)

    def decode_img_cv2(self, sample_num):
        # Required to Make Same Decoding at Train and Test Time
        sample_num_np = sample_num.numpy()
        fname = self.img_filenames[sample_num_np]
        img = cv2.imread(fname)[:, :, ::-1]
        dim = (self.params.img_width, self.params.img_height)
        img = cv2.resize(img, dim)
        img = np.array(img, dtype=np.float32)
        return img

    def decode_depth_cv2(self, sample_num):
        # Required to Make Same Decoding at Train and Test Time
        sample_num_np = sample_num.numpy()
        fname = self.depth_filenames[sample_num_np]
        depth = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
        depth = np.minimum(depth, self.params.maximum_depth_value)
        dim = (self.params.img_width, self.params.img_height)
        depth = cv2.resize(depth, dim)
        depth = np.array(depth, dtype=np.float32)
        depth = (depth*255.)/(self.params.maximum_depth_value)  # Depth in (0-255)
        depth = np.expand_dims(depth, axis=-1)
        depth = np.tile(depth, (1, 1, 3))
        return depth

    

    