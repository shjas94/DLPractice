import numpy as np
import tensorflow as tf

#### Standard SENet ####


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channel_in, ratio, **kwargs):
        super().__init__(**kwargs)
        self.gp = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(
            channel_in//ratio, kernel_initializer='he_normal', use_bias=False)
        self.act = tf.nn.swish()
        self.dense2 = tf.keras.layers.Dense(
            channel_in, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.reshape = tf.keras.layers.Reshape([1, 1, channel_in])
        self.mul = tf.keras.layers.multiply()

    def call(self, x):
        h = self.gp(x)
        h = self.dense1(h)
        h = self.act(h)
        h = self.dense2(h)
        h = self.reshape(h)
        return self.mul([x, h])


class SEResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channel_in, channel_out, kernel_size, reduction_ratio=16, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.ac1 = tf.nn.swish()
        self.conv1 = tf.keras.layers.Conv2D(
            channel_out, kernel_size=kernel_size, kernel_regularizer=regularizer, kernel_initializer='he_normal', padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.ac2 = tf.nn.swish()
        self.conv2 = tf.keras.layers.Conv2D(
            channel_out, kernel_size=kernel_size, kernel_regularizer=regularizer, kernel_initializer='he_normal', padding='same')
        self.se = SEBlock(channel_out, reduction_ratio)
        if channel_in == chennel_out:
            self.identity = lambda x: x
        else:
            self.identity = tf.keras.layers.Conv2D(
                channel_out, (1, 1), padding='same')

    def call(self, x, training=False):
        h = self.bn1(x, training=training)
        h = self.ac1(h)
        h = self.conv1(h)
        h = self.bn2(h, training=training)
        h = self.ac2(h)
        h = self.conv2(h)
        h = self.se(h)
        return self.identity(x) + h


class SEDeepResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channel_in, channel_out, kernel_size, regularizer=None, reduction_ratio=16, bottleneck_ratio=0.4, **kwargs):
        super().__init__(**kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.ac1 = tf.nn.swish()
        self.conv1 = tf.keras.layers.Conv2D(tf.math.floor(channel_out*bottleneck_ratio), kernel_size=(
            1, 1), kernel_regularizer=regularizer, kernel_initializer='he_normal', padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.ac2 = tf.nn.swish()
        self.conv2 = tf.keras.layers.Conv2D(tf.math.floor(channel_out*bottleneck_ratio), kernel_size=kernel_size,
                                            kernel_regularizer=regularizer, kernel_initializer='he_normal', padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.ac3 = tf.nn.swish()
        self.conv3 = tf.keras.layers.Conv2D(channel_out, kernel_size=(
            1, 1), kernel_regularizer=regularizer, kernel_initializer='he_normal', padding='same')
        self.se = SEBlock(channel_out, reduction_ratio)
        if channel_in == channel_out:
            self.identity = lambda x: x
        else:
            self.identity = tf.keras.layers.Conv2D(channel_out)

    def call(self, x, training=False):
        h = self.bn1(x, training=training)
        h = self.ac1(h)
        h = self.conv1(h)
        h = self.bn2(h, training=training)
        h = self.ac2(h)
        h = self.conv2(h)
        h = self.bn3(h, training=training)
        h = self.ac3(h)
        h = self.conv3(h)
        h = self.se(h)
        return self.identity(x) + h
