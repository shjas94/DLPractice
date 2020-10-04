import tensorflow as tf
import numpy as np


class DotScaledAttention(tf.keras.layers.Layer):
    def __init__(self, d_emb, d_reduced, masked=False):
        super().__init__()
        self.q = tf.keras.layers.Dense(d_reduced, input_shape=(-1, d_emb))
        self.k = tf.keras.layesr.Dense(d_reduced, input_shape=(-1, d_emb))
        self.v = tf.keras.layers.Dense(d_reduced, input_shape=(-1, d_emb))
        self.scale = tf.keras.layers.Lambda(lambda x: x/np.sqrt(d_reduced))
        self.masked = masked

    def call(self, x, training=None, mask=None):
        q = self.scale(self.q(x[0]))
        k = self.k(x[1])
        v = self.v(x[2])
        k_T = tf.transpose(k, perm=[0, 2, 1])
        comp = tf.matmul(q, k_T)
        if self.masked:
            length = tf.shape(comp)[-1]
            mask = tf.fill((length, length), -np.inf)
            mask = tf.linalg.band_part(mask, 0, -1)
            mask = tf.linalg.set_diag(mask, tf.zeros((length)))
            comp += mask
        comp = tf.nn.softmax(comp, axis=-1)
        return tf.matmul(comp, v)
