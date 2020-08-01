import numpy as np
import tensorflow as tf


class InceptionModule(tf.keras.Model):
    def __init__(self, filter_outs, activation='relu'):  # 각 size별 출력 list로
        super(InceptionModule, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(filter_outs[0], kernel_size=(
            1, 1), padding='same', activation=activation)
        self.conv3 = tf.keras.layers.Conv2D(filter_outs[1], kernel_size=(
            3, 3), padding='same', activation=activation)

        self.conv1_2 = tf.keras.layers.Conv2D(filter_outs[2], kernel_size=(
            1, 1), padding='same', activation=activation)
        self.conv5 = tf.keras.layers.Conv2D(filter_outs[3], kernel_size=(
            5, 5), padding='same', activation=activation)

        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(
            3, 3), strides=(1, 1), padding='same', activation=activation)
        self.conv1_3 = tf.keras.layers.Conv2D(filter_outs[4], kernel_size=(
            1, 1), padding='same', activation=activation)

        self.conv1_4 = tf.keras.layers.Conv2D(filter_outs[5], kernel_size=(
            1, 1), padding='same', activation=activation)

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        op1 = self.conv1_1(x)
        op1 = self.conv3(x)

        op2 = self.conv1_2(x)
        op2 = self.conv5(x)

        op3 = self.maxpool(x)
        op3 = self.conv1_3(x)

        op4 = self.conv1_4(x)

        return self.concat([op1, op2, op3, op4])
