import os
import cv2
import math as mt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
 
try:
    # Ros Runtime
    from .net import Network
    from .loader import Dataset
    from .utils import MixtureSpaceLoss, TrajectoryCostLoss
except:
    # Training Time
    from net import Network
    from loader import Dataset
    from utils import MixtureSpaceLoss, TrajectoryCostLoss

class Learner():
    def __init__(self, params):
        self.params = params
        self.network = Network(self.params)

        physical_devices = tf.config.experimental.list_physical_devices(self.params.devices)
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass

        self.min_val_loss = tf.Variable(np.inf, name='min_val_loss', trainable=False)
        self.space_loss = MixtureSpaceLoss(self.params, modes=self.params.modes)

        # Need Two Instances Due to Pointclouds
        self.cost_loss_tr = TrajectoryCostLoss(self.params, ref_frame=self.params.ref_frame, state_dim=self.params.state_dim)
        self.cost_loss_vl = TrajectoryCostLoss(self.params, ref_frame=self.params.ref_frame, state_dim=self.params.state_dim)

        # Rate Scheduler
        self.learning_rate_fn = tf.keras.experimental.CosineDecayRestarts(1e-3, 50000, 1.5, 0.75, 0.01)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)
        self.train_space_loss = tf.keras.metrics.Mean(name='train_space_loss')
        self.val_space_loss = tf.keras.metrics.Mean(name='validation_space_loss')
        self.train_cost_loss = tf.keras.metrics.Mean(name='train_cost_loss')
        self.val_cost_loss = tf.keras.metrics.Mean(name='validation_cost_loss')
        self.global_epoch = tf.Variable(0)
        self.ckpt = tf.train.Checkpoint(step=self.global_epoch, optimizer=self.optimizer, net=self.network)

        if self.params.resume_training:
            if self.ckpt.restore(self.params.resume_ckpt_file):
                print("------------------------------------------")
                print("Restored from {}".format(self.params.resume_ckpt_file))
                print("------------------------------------------")
                return

        print("------------------------------------------")
        print("Initializing from Scratch.")
        print("------------------------------------------")

    def train(self):
        # Training Procedure
        print('Training Network')
        self.summary_writer = tf.summary.create_file_writer(self.params.log_dir)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.params.log_dir, max_to_keep=20)

        dataset_train = Dataset(self.params, self.params.train_dir, is_training=True)
        dataset_val = Dataset(self.params, self.params.val_dir, is_training=False)

        # Add Pointclouds to Losses
        self.cost_loss_tr.add_pointclouds(dataset_train.pointclouds)
        self.cost_loss_vl.add_pointclouds(dataset_val.pointclouds)

        for epoch in range(self.params.training_epochs):
            # Perform a Train Step
            for k, (features, label, _) in enumerate(tqdm(dataset_train.batched_dataset)):
                features = self.adapt_input_data(features)
                gradients = self.train_step(features, label)

                if tf.equal(k % self.params.summary_freq, 0):
                    self.write_train_summaries(features, gradients)
                    self.train_space_loss.reset_states()
                    self.train_cost_loss.reset_states()

            # Evaluate Current Parameters
            for k, (features, label, _) in enumerate(tqdm(dataset_val.batched_dataset)):
                features = self.adapt_input_data(features)
                self.val_step(features, label)

            val_space_loss = self.val_space_loss.result()
            val_cost_loss = self.val_cost_loss.result()
            validation_loss = val_space_loss + val_cost_loss

            with self.summary_writer.as_default():
                tf.summary.scalar("Validation Space Loss", val_space_loss, step=tf.cast(self.global_epoch, dtype=tf.int64))
                tf.summary.scalar("Validation Cost Loss", val_cost_loss, step=tf.cast(self.global_epoch, dtype=tf.int64))

            self.val_space_loss.reset_states()
            self.val_cost_loss.reset_states()

            self.global_epoch = self.global_epoch + 1
            self.ckpt.step.assign_add(1)

            print("Epoch: {:2d}, Val Space Loss: {:.4f}, Val Cost Loss: {:.4f}".format(self.global_epoch, val_space_loss, val_cost_loss))

            if validation_loss < self.min_val_loss or ((epoch+1) % self.params.save_every_n_epochs) == 0:
                if validation_loss < self.min_val_loss:
                    self.min_val_loss = validation_loss
                if validation_loss < 10.0: # Otherwise Training Diverged
                    save_path = self.ckpt_manager.save()
                    print("Saved checkpoint for epoch {}: {}".format(int(self.ckpt.step), save_path))

        print("------------------------------")
        print("Training Finished Successfully")
        print("------------------------------")

    def write_train_summaries(self, features, gradients):
        with self.summary_writer.as_default():
            tf.summary.scalar('Train Space Loss', self.train_space_loss.result(), step=self.optimizer.iterations)
            tf.summary.scalar('Train Trajectory Cost Loss', self.train_cost_loss.result(), step=self.optimizer.iterations)

    def adapt_input_data(self, features):
        if self.params.use_rgb and self.params.use_depth:
            inputs = {"rgb": features[1][0],
                      "depth": features[1][1],
                      "roll_id": features[2],
                      "imu": features[0]}
        elif self.params.use_rgb and (not self.params.use_depth):
            inputs = {"rgb": features[1],
                      "roll_id": features[2],
                      "imu": features[0]}
        elif self.params.use_depth and (not self.params.use_rgb):
            inputs = {"depth": features[1],
                      "roll_id": features[2],
                      "imu": features[0]}
        return inputs

    def test(self):
        print("Testing Network")
        train_log_dir = os.path.join(self.params.log_dir, 'Test')
        dataset_eval = Dataset(self.params, self.params.test_dir, is_training=False)

        if not os.path.isdir(train_log_dir):
            os.makedirs(train_log_dir)

        idx_img = 0
        for features, label, traj_num in tqdm(dataset_eval.batched_dataset):
            features = self.adapt_input_data(features)
            alphas, predictions = self.inference(features)

            ref_img = features['depth'].numpy()[0, 0, :, :, 0]
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            
            predictions = np.array(predictions, dtype=float)
            predictions = predictions.reshape((self.params.state_dim, self.params.out_seq_len, self.params.modes))
            d_max = np.max(predictions[0, :, 0])
            for kk in range(self.params.out_seq_len):
                # for ii in range(self.params.modes):
                x = predictions[0, kk, 0]
                y = predictions[1, kk, 0]
                z = predictions[2, kk, 0]
                
                u = mt.ceil(-(y*self.params.fu/x) + self.params.cu)
                v = mt.ceil(-(z*self.params.fv/x) + self.params.cu)

                out_img = cv2.circle(ref_img, (u,v), radius=2, color=(255 - (x*255/d_max), 0, (x*255/d_max)), thickness=2)

            cv2.imwrite(train_log_dir + '/pred_{}.png'.format(idx_img), out_img)
            idx_img += 1

    def inference(self, u):
        # Run Time Inference
        prediction = self.net_inference(u).numpy()
        prediction = prediction[:, np.abs(prediction[0, :, 0]).argsort(), :]
        alphas = np.abs(prediction[0, :, 0])
        predictions = prediction[0, :, 1:]
        return alphas, predictions

    @tf.function
    def net_inference(self, u):
        return self.network(u, False)

    @tf.function
    def train_step(self, u, l):
        with tf.GradientTape() as tape:
            predictions = self.network(u, True)
            space_loss = self.space_loss(l, predictions)
            cost_loss = self.cost_loss_tr((u['roll_id'], u['imu'][:, -1, :12]), predictions)
            loss = space_loss + cost_loss

        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1) for g in gradients]

        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.train_space_loss.update_state(space_loss)
        self.train_cost_loss.update_state(cost_loss)
        return gradients

    @tf.function
    def val_step(self, u, l):
        predictions = self.network(u, False)
        space_loss = self.space_loss(l, predictions)
        cost_loss = self.cost_loss_vl((u['roll_id'], u['imu'][:, -1, :12]), predictions)

        self.val_space_loss.update_state(space_loss)
        self.val_cost_loss.update_state(cost_loss)
        return predictions