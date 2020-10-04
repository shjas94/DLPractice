import numpy as np
import tensorflow as tf


# Referred from https://github.com/LastRemote/Transformer-TF2.0
class PositionalEncoding(Layer):
    def __init__(self, max_len, d_emb):
        super().__init__()
        self.sinusoidal_encoding = np.array([self.get_positional_angle(
            pos, d_emb) for pos in range(max_len)], dtype=np.float32)
        self.sinusoidal_encoding[:, 0::2] = np.sin(
            self.sinusoidal_encoding[:, 0::2])
        self.sinusoidal_encoding[:, 1::2] = np.cos(
            self.sinusoidal_encoding[:, 1::2])
        self.sinusoidal_encoding = tf.cast(
            self.sinusoidal_encoding, dtype=tf.float32)

    def call(self, x, training=None, mask=None):
        return x + self.sinusoidal_encoding[:tf.shape(x)[1]]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_angle(self, pos, dim, d_emb):
        return pos / np.power(10000, 2 * (dim // 2) / d_emb)

    def get_positional_angle(self, pos, d_emb):
        return [self.get_angle(pos, dim, d_emb) for dim in range(d_emb)]
