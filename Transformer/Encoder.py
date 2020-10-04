import tensorflow as tf
from Transformer.MultiHeadAttention import MultiHeadAttention


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_head, d_reduced):
        super().__init__()
        self.num_head = num_head
        self.d_r = d_reduced

    def build(self, input_shape):
        self.multi_attention = MultiHeadAttention(
            self.num_head, input_shape[-1], self.d_r)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(
            input_shape=input_shape)
        self.dense1 = tf.keras.layers.Dense(
            input_shape[-1] * 4, input_shape=input_shape, activation='relu')
        self.dense2 = tf.keras.layers.Dense(
            input_shape[-1], input_shape=self.dense1.compute_output_shape(input_shape))
        self.layer_norm2 = tf.keras.layers.LayerNormalization(
            input_shape=input_shape)
        super().build()

    def call(self, x, training=None, mask=None):
        h = self.multi_attention((x, x, x))
        ln1 = self.layer_norm1(x + h)  # norm + skipConnection

        h = self.dense2(self.dense1(ln1))
        return self.layer_norm2(h + ln1)

    def compute_output_shape(self, input_shape):
        return input_shape
