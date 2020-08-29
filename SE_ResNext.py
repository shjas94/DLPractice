import tensorflow as tf


class ResNextBlock(tf.keras.layers.Layer):
    def __init__(self, filter_in, filter_out, kernel_size, cardinality=32, **kwargs):
        super().__init__(**kwargs)
        ##HyperParameter##
        self.filter_in = filter_in
        self.filter_out = filter_out
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.cardinality = cardinality
        ##################
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.elu1 = tf.keras.layers.ELU()
        self.conv1 = tf.keras.layers.Conv2D(
            filter_in//2, (1, 1), padding='same', kernel_initializer="he_normal", kernel_constraint=tf.keras.constraints.max_norm(3.))

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.elu2 = tf.keras.layers.ELU()
        self.conv2 = tf.keras.layers.Conv2D(
            filter_in//2, kernel_size, padding='same', kernel_initializer="he_normal", group=cardinality, kernel_constraint=tf.keras.constraints.max_norm(3.))

        self.bn3 = tf.keras.layers.BatchNormalization()
        self.elu3 = tf.keras.layers.ELU()
        self.conv3 = tf.keras.layers.Conv2D(
            filter_out, (1, 1), padding='same', kernel_initializer="he_normal", kernel_constraint=tf.keras.constraints.max_norm(3.))

    def call(self, x, training=None):
        h = self.bn1(x, training=training)
        h = self.elu1(h)
        h = self.conv1(h)

        h = self.bn2(h, training=training)
        h = self.elu2(h)
        h = self.conv2(h)

        h = self.bn3(h, training=training)
        h = self.elu3(h)
        return self.conv3(h)

    def get_config(self):
        config = super().get_config()
        config.update({"filter_in": self.filter_in, "filter_out": self.filter_out,
                       "cardinality": self.cardinality, "kernel_size": self.kernel_size})
        return config


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, filter_in, reduction_ratio, **kwargs):
        super().__init__(**kwargs)
        ##HyperParameter##
        self.filter_in = filter_in
        self.reduction_ratio = reduction_ratio
        ##################
        self.gp = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(
            filter_in//reduction_ratio, kernel_initializer="he_normal", activation='elu', use_bias=False)
        self.dense2 = tf.keras.layers.Dense(
            filter_in, activation='sigmoid', kernel_initializer="he_normal", use_bias=False)
        self.reshape = tf.keras.layers.Reshape([1, 1, filter_in])
        self.mul = tf.keras.layers.Multiply()

    def call(self, x):
        h = self.gp(x)
        h = self.dense1(h)
        h = self.dense2(h)
        h = self.reshape(h)
        return self.mul([x, h])

    def get_config(self):
        config = super().get_config()
        config.update({'filter_in': self.filter_in,
                       'reduction_ratio': self.reduction_ratio})
        return config


class SeResNextLayer(tf.keras.layers.Layer):
    def __init__(self, filter_in, filter_out, reduction_ratio, kernel_size, cardinality=32, **kwargs):
        super().__init__(**kwargs)
        ##HyperParameter##
        self.filter_in = filter_in
        self.filter_out = filter_out
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.cardinality = cardinality
        ##################
        self.rexmodule = ResNextBlock(
            filter_in, filter_out, kernel_size, cardinality)
        self.add = tf.keras.layers.Add()
        self.se = SEBlock(filter_out, reduction_ratio)
        self.identity = tf.keras.layers.Conv2D(
            filter_out, (1, 1), padding='same')

    def call(self, x, training=None):
        h = self.rexmodule(x)
        h = self.se(h)
        return self.identity(x) + h

    def get_config(self):
        config = super().get_config()
        config.update({'filter_in': self.filter_in, 'filter_out': self.filter_out, 'reduction_ratio': self.reduction_ratio,
                       'kernel_size': self.kernel_size, 'cardinality': self.cardinality})
        return config
