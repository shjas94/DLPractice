import tensorflow as tf
from Transformer.Encoder import Encoder
from Transformer.Decoder import Decoder
from Transformer.PositionalEncoding import PositionalEncoding


class Transformer(tf.keras.Model):
    def __init__(self, src_vocab, dst_vocab, max_len, d_emb, d_reduced, n_enc_layer, n_dec_layer, num_head):
        super().__init__()
        self.enc_emb = tf.keras.layers.Embedding(src_vocab, d_emb)
        self.dec_emb = tf.keras.layers.Embedding(dst_vocab, d_emb)
        self.pos_enc = PositionalEncoding(max_len, d_emb)

        self.encoder = [Encoder(num_head, d_reduced)
                        for _ in range(n_enc_layer)]
        self.decoder = [Decoder(num_head, d_reduced)
                        for _ in range(n_dec_layer)]
        self.dense = tf.keras.layers.Dense(dst_vocab, input_shape=(-1, d_emb))

    def call(self, inputs, training=None, mask=None):
        src_sent, dst_sent_shifted = inputs

        h_enc = self.pos_enc(self.enc_emb(src_sent))
        for enc in self.encoder:
            h_enc = enc(h_enc)

        h_dec = self.pos_enc(self.dec_emb(dst_sent_shifted))
        for dec in self.decoder:
            h_dec = dec([h_dec, h_enc])

        return tf.nn.softmax(self.dense(h_dec), axis=-1)
