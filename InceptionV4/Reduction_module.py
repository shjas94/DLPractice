import tensorflow as tf


class Reduction_A(tf.keras.layers.Layer):
    def __init__(self, k, l, m, n, **kwargs):
        super().__init__(**kwargs)
        ## HyperParameters##
        self.k = k
        self.l = l
        self.m = m
        self.n = n
        ####################

        self.pool1 = tf.keras.layers.MaxPool2D(
            (3, 3), strides=2, padding='valid')

        self.conv2 = tf.keras.layers.Conv2D(self.n, kernel_size=(3, 3), strides=(
            2, 2), padding='valid', activation='elu', kernel_initializer='he_normal')

        self.conv3_1 = tf.keras.layers.Conv2D(self.k, kernel_size=(
            1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv3_2 = tf.keras.layers.Conv2D(self.l, kernel_size=(
            3, 3), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv3_3 = tf.keras.layers.Conv2D(self.m, kernel_size=(
            3, 3), padding='valid', strides=2, activation='elu', kernel_initializer='he_normal')

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x):

        h1 = self.pool1(x)

        h2 = self.conv2(x)

        h3 = self.conv3_1(x)
        h3 = self.conv3_2(h3)
        h3 = self.conv3_3(h3)

        return self.concat([h1, h2, h3])

    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k, "l": self.l, "m": self.m,
                       "n": self.n})
        return config


class Reduction_B(tf.keras.layers.Layer):
    def __init__(self, filter_outputs, **kwargs):
        super().__init__(**kwargs)
        ## HyperParameters##
        self.filter_outputs = filter_outputs
        ####################

        self.pool = tf.keras.layers.MaxPool2D(
            kernel_size=(3, 3), strides=2, padding='valid')

        self.conv2_1 = tf.keras.layers.Conv2D(self.filter_outputs[0], kernel_size=(
            1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv2_2 = tf.keras.layers.Conv2D(self.filter_outputs[0], kernel_size=(
            3, 3), padding='valid', strides=2, activation='elu', kernel_initializer='he_normal')

        self.conv3_1 = tf.keras.layers.Conv2D(self.filter_outputs[1]-100, kernel_size=(
            1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv3_2 = tf.keras.layers.Conv2D(self.filter_outputs[1]-100, kernel_size=(
            1, 7), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv3_3 = tf.keras.layers.Conv2D(self.filter_outputs[1], kernel_size=(
            7, 1), padding='same', activation='elu', kernel_initializer='he_normal')
        self.conv3_4 = tf.keras.layers.Conv2D(self.filter_outputs[1], kernel_size=(
            3, 3), padding='valid', strides=2, activation='elu', kernel_initializer='he_normal')

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        h1 = self.pool(x)

        h2 = self.conv2_1(x)
        h2 = self.conv2_2(h2)

        h3 = self.conv3_1(x)
        h3 = self.conv3_2(h3)
        h3 = self.conv3_3(h3)
        h3 = self.conv3_4(h3)

        return self.concat([h1, h2, h3])

    def get_config(self):
        config = super().get_config()
        config.update({"filter_outputs": self.filter_outputs})
        return config
