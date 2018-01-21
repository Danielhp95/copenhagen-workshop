import tensorflow as tf
from pysc2.lib.features import SCREEN_FEATURES
from tensorflow.contrib import layers
from baselines.a2c.utils import sample

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample
import numpy as np


class FullyConvPolicy:
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        X = tf.placeholder(tf.int32, ob_shape)  # obs

        with tf.variable_scope("fullyconv_model", reuse=reuse):
            x_onehot = layers.one_hot_encoding(
                # assuming we have only one channel
                X[:, :, :, 0],
                num_classes=SCREEN_FEATURES.player_relative.scale
            )

            #don't one hot 0-category
            x_onehot = x_onehot[:, :, :, 1:]

            h = layers.conv2d(
                x_onehot,
                num_outputs=16,
                kernel_size=5,
                stride=1,
                padding='SAME',
                scope="conv1"
            )
            h2 = layers.conv2d(
                h,
                num_outputs=32,
                kernel_size=3,
                stride=1,
                padding='SAME',
                scope="conv2"
            )
            pi = layers.flatten(layers.conv2d(
                h,
                num_outputs=1,
                kernel_size=1,
                stride=1,
                scope="spatial_action",
                activation_fn=None
            ))

            pi *= 3.0  # make it little bit more deterministic, not sure if good idea

            f = layers.fully_connected(
                layers.flatten(h2),
                num_outputs=64,
                activation_fn=tf.nn.relu,
                scope="value_h_layer"
            )

            vf = layers.fully_connected(
                f,
                num_outputs=1,
                activation_fn=None,
                scope="value_out"
            )

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v, []  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
