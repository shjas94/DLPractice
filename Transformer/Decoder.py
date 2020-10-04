import tensorflow as tf
from Transformer.MultiHeadAttention import MultiHeadAttention


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_head, d_reduced):
        super().__init__()
        self.num_head = num_head
        self.d_r = d_reduced

    def build(self, input_shape):
        self.self_attention = MultiHeadAttention(
            self.num_head, input_shape[0][-1], self.d_r, masked=True)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(
            input_shape=input_shape)

        self.multi_attention = MultiHeadAttention(
            self.num_head, input_shape[0][-1], self.d_r)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(
            input_shape=input_shape)

        self.dense1 = tf.keras.layers.Dense(
            input_shape[0][-1] * 4, input_shape=input_shape[0], activation='relu')
        self.dense2 = tf.keras.layers.Dense(
            input_shape[0][-1], input_shape=self.dense1.compute_output_shape(input_shape[0]))
        self.layer_norm3 = tf.keras.layers.LayerNormalization(
            input_shape=input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x, context = inputs
        h = self.self_attention((x, x, x))
        ln1 = self.layer_norm1(x + h)

        h = self.multi_attention((ln1, context, context))
        ln2 = self.layer_norm2(ln1 + h)

        h = self.dense2(self.dense1(ln2))
        return self.layer_norm3(h + ln2)

    def compute_output_shape(self, input_shape):
        return input_shape
