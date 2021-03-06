import numpy as np
import tensorflow as tf


class DenseUnit(tf.keras.layers.Layer):
    def __init__(self, filter_out, kernel_size):
        super(DenseUnit, self).__init__()
        self.filter_out = filter_out
        self.kernel_size = kernel_size
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(
            self.filter_out, self.kernel_size, padding='same')
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False, mask=None):  # x: (Batch, H, W, Ch_in)
        h = self.bn(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv(h)  # h: (Batch, H, W, filter_out)
        return self.concat([x, h])  # (Batch, H, W, (Ch_in, filter_output))

    def get_config(self):
        config = super().get_config()
        config.update({"filter_out": self.filter_out,
                       "kernel_size": self.kernel_size})
        return config


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_unit, growth_rate, kernel_size):
        super(DenseLayer, self).__init__()
        self.num_unit = num_unit
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.sequence = list()
        for idx in range(num_unit):
            self.sequence.append(DenseUnit(growth_rate, kernel_size))

    def call(self, x, training=False, mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"num_unit": self.num_unit, "growth_rate": self.growth_rate,
                       "kernel_size": self.kernel_size})
        return config


class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(TransitionLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same')
        self.pool = tf.keras.layers.MaxPool2D()

    def call(self, x, training=False, mask=None):
        x = self.conv(x)
        return self.pool(x)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters,
                       "kernel_size": self.kernel_size})

# class DenseNet(tf.keras.Model):
#     def __init__(self):
#         super(DenseNet, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(
#             8, (3, 3), padding='same', activation='relu')  # %mnist -> 28x28x8
#         self.dl1 = DenseLayer(2, 4, (3, 3))  # 28x28x16
#         self.tr1 = TransitionLayer(16, (3, 3))  # 14x14x16

#         self.dl2 = DenseLayer(2, 8, (3, 3))  # 14x14x32
#         self.tr2 = TransitionLayer(32, (3, 3))  # 7*7*32

#         self.dl3 = DenseLayer(2, 16, (3, 3))  # 7x7x64

#         self.flatten = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(128, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

#     def build(self, input_shape):
#         super().build(input_shape)

#     def call(self, x, training=False, mask=None):
#         x = self.conv1(x)
#         x = self.dl1(x, training=training)
#         x = self.tr1(x
#                      )
#         x = self.dl2(x, training=training)
#         x = self.tr2(x)

#         x = self.dl3(x, training=training)

#         x = self.flatten(x)
#         x = self.dense1(x)
#         return self.dense2(x)
