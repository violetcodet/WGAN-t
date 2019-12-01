import tensorflow as tf
import tensorflow.contrib.layers as tcl

from layers import *


class Discriminator(object):
    def __init__(self):
        self.x_dim = 512 * 512 * 3
        self.name = 'lsun/dcgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 512, 512, 3])
            conv1 = tcl.conv2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu
            )
            conv2 = tcl.conv2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv3 = tcl.conv2d(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = tcl.conv2d(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv5 = tcl.conv2d(
                conv4, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv6 = tcl.conv2d(
                conv5, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv7 = tcl.conv2d(
                conv6, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv7 = tcl.flatten(conv7)
            fc = tcl.fully_connected(conv7, 1, activation_fn=tf.identity)
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 500 * 500 * 3
        self.name = 'lsun/dcgan/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            bs = tf.shape(z)[0]
            fc = tcl.fully_connected(z, 4 * 4 * 1024, activation_fn=tf.identity)
            conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, 1024]))
            conv1 = relu_batch_norm(conv1)
            conv2 = tcl.conv2d_transpose(
                conv1, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv3 = tcl.conv2d_transpose(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv4 = tcl.conv2d_transpose(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv5 = tcl.conv2d_transpose(
                conv4, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv6 = tcl.conv2d_transpose(
                conv5, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv7 = tcl.conv2d_transpose(
                conv6, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv8 = tcl.conv2d_transpose(
                conv7, 3, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.tanh)
            return conv8

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]