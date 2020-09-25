import tensorflow as tf
from InceptionV4_stem import stemLayer
from Inception_module import Inception_A, Inception_B, Inception_C
from Reduction_module import Reduction_A, Reduction_B


class InceptionV4(tf.keras.Model):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__(**kwargs)
        ## HyperParameters##
        self.input_shape = input_shape
        self.output_shape = output_shape
        ####################
        self.stem = stemLayer(self.input_shape)
        self.inception_a = [Inception_A(384) for _ in range(4)]
        self.reduction_a = Reduction_A(192, 224, 256, 384)
        self.inception_b = [Inception_B(1024) for _ in range(7)]
        self.reduction_b = Reduction_B(1536)
        self.inception_c = [Inception_C(1536) for _ in range(3)]
        self.gbavgpool = tf.keras.layers.AveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.8)
        self.clf = tf.keras.layers.Dense(
            self.output_shape, activaion='softmax')

    def call(self, x):
        h = self.stem(x)
        for a in self.inception_a:
            h = a(h)
        h = self.reduction_a(x)
        for b in self.inception_b:
            h = b(h)
        h = self.reduction_b(h)
        for c in self.inception_c:
            h = c(h)
        h = self.gbavgpool(h)
        h = self.dropout(h)
        return self.clf(h)

    def get_config(self):
        config = super().get_config()
        config.update({"input_shape": self.input_shape,
                       "output_shape": self.output_shape})
        return config
