import numpy as np
import tensorflow as tf


class InceptionModule(tf.keras.Model):
    ## [(conv1_1, conv3), (conv1_2, conv5), conv1_3, conv1_4]
    def __init__(self, filter_outs, activation='relu'):  # 각 size별 출력 list로
        super(InceptionModule, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(filter_outs[0][0], kernel_size=(
            1, 1), padding='same', activation=activation)
        self.conv3 = tf.keras.layers.Conv2D(filter_outs[0][1], kernel_size=(
            3, 3), padding='same', activation=activation)

        self.conv1_2 = tf.keras.layers.Conv2D(filter_outs[1][0], kernel_size=(
            1, 1), padding='same', activation=activation)
        self.conv5 = tf.keras.layers.Conv2D(filter_outs[1][1], kernel_size=(
            5, 5), padding='same', activation=activation)

        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(
            3, 3), strides=(1, 1), padding='same', activation=activation)
        self.conv1_3 = tf.keras.layers.Conv2D(filter_outs[2], kernel_size=(
            1, 1), padding='same', activation=activation)

        self.conv1_4 = tf.keras.layers.Conv2D(filter_outs[3], kernel_size=(
            1, 1), padding='same', activation=activation)

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        op1 = self.conv1_1(x)
        op1 = self.conv3(op1)

        op2 = self.conv1_2(x)
        op2 = self.conv5(op2)

        op3 = self.maxpool(x)
        op3 = self.conv1_3(op3)

        op4 = self.conv1_4(x)

        return self.concat([op1, op2, op3, op4])


class GoogleNet(tf.keras.Model):
    def __init__(self, activation='relu'):
        super(GoogleNet, self).__init__()  # cifar-10
        self.conv1 = tf.keras.layers.Conv2D(
            8, (5, 5), padding='same', activation=activation)  # 32x32x8
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2))  # 16x16x8
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            16, (1, 1), padding='same', activation=activation)  # 16x16x16
        self.conv3 = tf.keras.layers.Conv2D(
            32, (3, 3), padding='same', activation=activation)  # 16x16x32
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2))  # 8x8x32
        self.inception1 = InceptionModule(
            [(32, 64), (8, 16), 16, 32], activation='relu')
        self.inception2 = InceptionModule(
            [(24, 48), (6, 12), 16, 30], activation='relu')
        self.inception3 = InceptionModule(
            [(20, 40), (6, 12), 16, 48], activation='relu')
        self.inception4 = InceptionModule(
            [(16, 32), (4, 8), 16, 64], activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))  # 4x4
        self.inception5 = InceptionModule(
            [(8, 16), (4, 16), 16, 48], activation='relu')
        self.pool4 = tf.keras.layers.GlobalAveragePooling2D((3, 3))
        self.drop = tf.keras.layers.Dropout(0.4)
        self.dense1 = tf.keras.layers.Dense(256, activation=activation)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.bn1(h, training=training)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.bn2(h, training=training)
        h = self.pool2(h)

        h = self.inception1(h)
        h = self.inception2(h)
        h = self.inception3(h)
        h = self.inception4(h)

        h = self.pool3(h)
        h = self.inception5(h)
        h = self.pool4(h)
        h = self.drop(h)
        h = self.dense1(h)
        return self.dense2(h)
