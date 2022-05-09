# Tensorflow Imports
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Conv2D, LeakyReLU, Conv1D
from keras.layers import Flatten, GlobalAveragePooling2D, MaxPool2D, LayerNormalization, BatchNormalization

# Mobilenet Imports
from keras.applications import mobilenet_v3
from keras.applications.mobilenet_v3 import preprocess_input

class Network(Model):
    def __init__(self, params):
        super(Network, self).__init__()
        self.params = params

        # Define all Network Layers
        self.img_backend = [mobilenet_v3.MobileNetV3Large(include_top=False,
                                                       weights='imagenet',
                                                       input_shape=self.params.input_size,
                                                       pooling=None)]
        
        if self.params.freeze_backend:
            self.img_backend[0].trainable = False
        else:
            # Freeze Only BN Blocks
            m = self.img_backend[0]
            for layer in m.layers:
                if layer.name.endswith('_bn'):
                    layer.trainable = False
                else:
                    layer.trainable = True
        
        # Reduce a Bit the Output Size
        self.resize_operator_1 = [Conv1D(128, kernel_size=1, strides=1, padding='valid', dilation_rate=1, use_bias=self.params.use_bias)]
        self.img_frontend = [Conv1D(int(128*self.params.g_im), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                             LeakyReLU(alpha=1e-2),
                             Conv1D(int(64*self.params.g_im), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                             LeakyReLU(alpha=1e-2),
                             Conv1D(int(64*self.params.g_im), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                             LeakyReLU(alpha=1e-2),
                             Conv1D(int(32*self.params.g_im), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                             LeakyReLU(alpha=1e-2)]
        self.resize_operator_2 = [Conv1D(self.params.modes, kernel_size=3, strides=1, padding='valid', dilation_rate=1, use_bias=self.params.use_bias)]

        # Odometry Layers
        self.odom_frontend = [Conv1D(int(64*self.params.g_ss), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                              LeakyReLU(alpha=.5),
                              Conv1D(int(32*self.params.g_ss), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                              LeakyReLU(alpha=.5),
                              Conv1D(int(32*self.params.g_ss), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                              LeakyReLU(alpha=.5),
                              Conv1D(int(32*self.params.g_ss), kernel_size=2, strides=1, padding='same', dilation_rate=1)]
        self.resize_operator_3 = [Conv1D(self.params.modes, kernel_size=3, strides=1, padding='valid', dilation_rate=1, use_bias=self.params.use_bias)]

        # Planning Layers
        output_dim = self.params.state_dim*self.params.out_seq_len + 1
        self.plan_frontend = [Conv1D(int(64*self.params.g_pl), kernel_size=1, strides=1, padding='valid'),
                              LeakyReLU(alpha=.5),
                              Conv1D(int(128*self.params.g_pl), kernel_size=1, strides=1, padding='valid'),
                              LeakyReLU(alpha=.5),
                              Conv1D(int(128*self.params.g_pl), kernel_size=1, strides=1, padding='valid'),
                              LeakyReLU(alpha=.5),
                              Conv1D(output_dim, kernel_size=1, strides=1, padding='same')]

    def call(self, u, training=False):
        # Implement Network's Forward Pass
        img_embeddings = self.img_prediction(u)
        imu_embeddings = self.imu_prediction(u)
        return self.plan_prediction(tf.concat((img_embeddings, imu_embeddings), axis=-1))

    def img_prediction(self, u):
        # Implement Images Forward Pass
        img = u['depth']                            # (batch_size, seq_len, img_height, img_width, 1)
        img = tf.transpose(img, (1, 0, 2, 3, 4))    # (seq_len, batch_size, img_height, img_width, 1)

        img_fts = tf.map_fn(self.conv_branch, elems=img, parallel_iterations=self.params.seq_len) # (seq_len, batch_size, modes, channels)

        # img_fts (seq_len, batch_size, MxMxC)
        img_fts = tf.transpose(img_fts, (1,0,2)) # batch_size, seq_len, MxMx128

        x = img_fts
        for f in self.img_frontend:
            x = f(x)

        # final x (batch_size, seq_len, 64)
        x = tf.transpose(x, (0,2,1)) # (batch_size, 64, seq_len)
        for f in self.resize_operator_2:
            x = f(x)

        # final x (batch_size, 64, modes)
        return tf.transpose(x, (0,2,1)) # (batch_size, modes, 64)

    def conv_branch(self, img):
        x = preprocess_input(img)
        for f in self.img_backend:
            x = f(x)
        
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1]))  # (batch_size, MxM, C)
        for f in self.resize_operator_1:
            x = f(x)

        # x [batch_size, M, M, 128]
        return tf.reshape(x, (x.shape[0], -1)) # (batch_size, MxMx128)

    def imu_prediction(self, u):
        x = u['imu'][:, :, 12:] # [B, seq_len, D]
        for f in self.odom_frontend:
            x = f(x)

        x = tf.transpose(x, (0,2,1)) # (batch_size, 32, seq_len)
        for f in self.resize_operator_3:
            x = f(x)

        # final x # [batch_size, 32, modes]
        return tf.transpose(x, (0, 2, 1)) # (batch_size, modes, 32)

    def plan_prediction(self, u):
        x = u
        for f in self.plan_frontend:
            x = f(x)

        return x