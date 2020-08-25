import numpy as np
import tensorflow as tf

# subclassing test


class SE_ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filter_in, filter_out, reduction_ratio, kernel_size, **kwargs):
        super().__init__(**kwargs)
        ##HyperParameter##
        self.filter_in = filter_in
        self.filter_out = filter_out
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        ##################
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.elu1 = tf.keras.layers.ELU()
        self.conv1 = tf.keras.layers.Conv2D(
            filter_out//4, (1, 1), padding='same', kernel_initializer="he_normal", kernel_constraint=tf.keras.constraints.max_norm(3.))

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.elu2 = tf.keras.layers.ELU()
        self.conv2 = tf.keras.layers.Conv2D(
            filter_out//4, kernel_size, padding='same', kernel_initializer="he_normal", kernel_constraint=tf.keras.constraints.max_norm(3.))

        self.bn3 = tf.keras.layers.BatchNormalization()
        self.elu3 = tf.keras.layers.ELU()
        self.conv3 = tf.keras.layers.Conv2D(
            filter_out, (1, 1), padding='same', kernel_initializer="he_normal", kernel_constraint=tf.keras.constraints.max_norm(3.))

        self.gp = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(
            filter_out//reduction_ratio, kernel_initializer="he_normal", activation='elu', use_bias=False)
        self.dense2 = tf.keras.layers.Dense(
            filter_out, activation='sigmoid', kernel_initializer="he_normal", use_bias=False)
        self.reshape = tf.keras.layers.Reshape([1, 1, filter_out])
        self.mul = tf.keras.layers.Multiply()

        self.identity = tf.keras.layers.Conv2D(
            filter_out, (1, 1), padding='same')

    def call(self, x, training=None):
        h = self.bn1(x, training=training)
        h = self.elu1(h)
        h = self.conv1(h)

        h = self.bn2(h, training=training)
        h = self.elu2(h)
        h = self.conv2(h)

        h = self.bn3(h, training=training)
        h = self.elu3(h)
        h = self.conv3(h)

        s = self.gp(h)
        s = self.dense1(s)
        s = self.dense2(s)
        s = self.reshape(s)
        s = self.mul([s, h])
        return self.identity(x) + s

    def get_config(self):
        config = super().get_config()
        config.update({"filter_in": self.filter_in, "filter_out": self.filter_out,
                       "reduction_ratio": self.reduction_ratio, "kernel_size": self.kernel_size})
        return config
