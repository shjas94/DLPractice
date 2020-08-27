import tensorflow as tf


# Stem 부분은 filter concatenate 기준으로 3 block으로 나눠서 구현
# 논문에서 표기된 부분 이외의 layer에서는 padding='valid', strides=(2,2)
class stemBlock1(tf.keras.layers.Layer):
    def __init__(self, strides=(2, 2), padding='valid', **kwargs):
        super().__init__(**kwargs)
        ## HyperParameters##
        self.strides = strides
        self.padding = padding
        ####################

        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=self.strides, padding=self.padding, activation='elu', kernel_initializer='he_normal')
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(
            1, 1), padding=self.padding, activation='elu', kernel_initializer='he_normal')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(
            1, 1), padding='same', activation='elu', kernel_initializer='he_normal')

        self.conv4 = tf.keras.layers.Conv2D(
            96, (3, 3), strides=self.strides, padding=self.padding, activation='elu', kernel_initializer='he_normal')
        self.pool = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=self.strides)

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        h1 = self.conv4(h)
        h2 = self.pool(h)

        return self.concat([h1, h2])

    def get_config(self):
        config = super().get_config()
        config.update({"strides": self.strides, "padding": self.padding})
        return config


class stemBlock2(tf.keras.layers.Layer):
    def __init__(self, padding='valid', **kwargs):
        ## HyperParameters##
        self.padding = padding
        ####################

        self.conv1_1 = tf.keras.layers.Conv2D(64, (1, 1), strides=(
            1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv1_2 = tf.keras.layers.Conv2D(64, (7, 1), strides=(
            1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv1_3 = tf.keras.layers.Conv2D(64, (1, 7), strides=(
            1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv1_4 = tf.keras.layers.Conv2D(96, (3, 3), strides=(
            1, 1), padding=self.padding, activation='elu', kernel_initializer='he_normal')

        self.conv2_1 = tf.keras.layers.Conv2D(64, (1, 1), strides=(
            1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv2_2 = tf.keras.layers.Conv2D(96, (3, 3), strides=(
            1, 1), padding=self.padding, activation='elu', kernel_initializer='he_normal')

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        h1 = self.conv1_1(x)
        h1 = self.conv1_2(h1)
        h1 = self.conv1_3(h1)
        h1 = self.conv1_4(h1)

        h2 = self.conv2_1(x)
        h2 = self.conv2_2(h2)

        return self.concat([h1, h2])

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config


class stemBlock3(tf.keras.layers.Layer):
    def __init__(self, strides=(2, 2), padding='valid'):
        ## HyperParameters##
        self.strides = strides
        self.padding = padding
        ####################

        self.conv = tf.keras.layers.Conv2D(192, strides=(
            1, 1), padding=padding, activation='elu', kernel_initializer='he_normal')
        self.pool = tf.keras.layers.MaxPool2D(strides=strides, padding=padding)
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        h1 = self.conv(x)
        h2 = self.pool(x)
        return self.concat([h1, h2])

    def get_config(self):
        config = super().get_config()
        config.update({"strides": self.strides, "padding": self.padding})
        return config


class stemLayer(tf.keras.layers.Layer):
    pass
