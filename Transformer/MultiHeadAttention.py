import tensorflow as tf
from Transformer.DotScaledAttention import DotScaledAttention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_head, d_emb, d_reduced, masked=False):
        super.__init__()
        self.attention_list = list()
        for _ in range(num_head):
            self.attention_list.append(
                DotScaledAttention(d_emb, d_reduced, masked))
        self.linear = Dense(d_emb, input_shape=(-1, num_head * d_reduced))

    def call(self, x, training=None, mask=None):
        attention_list = [a(x, training) for a in self.attention_list]
        concat = tf.concat(attention_list, axis=-1)
        return self.linear(concat)
