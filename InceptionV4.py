import tensorflow as tf

########### Stem Block ##########

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
        super().__init__(**kwargs)
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
    def __init__(self, strides=(2, 2), padding='valid', **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, strides=(2, 2), padding='valid', residual=False, *kwargs):
        super().__init__(**kwargs)
        ## HyperParameters##
        self.strides = strides
        self.padding = padding
        self.residual = residual
        ####################
        if self.residual:
            pass
        else:
            self.stem1 = stemBlock1(strides=self.strides, padding=self.padding)
            self.stem2 = stemBlock2(padding=self.padding)
            self.stem3 = stemBlock3(strides=self.strides, padding=self.padding)

    def call(self, x):
        if self.residual:
            pass
        else:
            h = self.stem1(x)
            h = self.stem2(h)
            return self.stem3(h)

    def get_config(self):
        config = super().get_config()
        config.update({"strides": self.strides,
                       "padding": self.padding, "residual": self.residual})
        return config


########## Inception Module ##########
class Inception_A(tf.keras.layers.Layer):
    # filter out은 전체 모듈의 출력 필터 수 아님, 각 줄기마다의 필터 수 리스트로 전달
    def __init__(self, filter_outs, residual=False, **kwargs):
        super().__init__(**kwargs)
        ## HyperParameters##
        self.filter_outs = filter_outs
        self.residual = residual
        ####################
        if residual:
            pass
        else:
            self.avgpool = tf.keras.layers.AveragePooling2D()
            self.conv1 = tf.keras.layers.Conv2D(self.filter_outs[0], kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')

            self.conv2 = tf.keras.layers.Conv2D(self.filter_out[1], kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')

            self.conv3_1 = tf.keras.layers.Conv2D((self.filter_out[2]//3)*2, kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv3_2 = tf.keras.layers.Conv2D(self.filter_out[2], kernel_size=(
                3, 3), padding='same', activation='elu', kernel_initializer='he_normal')

            self.conv4_1 = tf.keras.layers.Conv2D((self.filter_out[3]//3)*2, kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_2 = tf.keras.layers.Conv2D(self.filter_out[3], kernel_size=(
                3, 3), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_3 = tf.keras.layers.Conv2D(self.filter_out[3], kernel_size=(
                3, 3), padding='same', activation='elu', kernel_initializer='he_normal')

            self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        if self.residual:
            pass
        else:
            h1 = self.avgpool(x)
            h1 = self.conv1(h1)

            h2 = self.conv2(x)

            h3 = self.conv3_1(x)
            h3 = self.conv3_2(h3)

            h4 = self.conv4_1(x)
            h4 = self.conv4_2(h4)
            h4 = self.conv4_3(h4)

            return sefl.concat([h1, h2, h3, h4])

    def get_config(self):
        config = super().get_config()
        config.update({"filter_outs": self.filter_outs,
                       "residual": self.residual})
        return config


class Inception_B(tf.keras.layers.Layer):
    def __init__(self, filter_outs, residual=False, **kwargs):
        super().__init__(**kwargs)
        ## HyperParameters##
        self.filter_outs = filter_outs
        self.residual = residual
        ####################
        if residual:
            pass
        else:
            self.avgpool = tf.keras.layers.AveragePooling2D()
            self.conv1_1 = tf.keras.layers.Conv2D(self.filter_outs[0], kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')

            self.conv2_1 = tf.keras.layers.Conv2D(self.filter_outs[1], kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')

            self.conv3_1 = tf.keras.layers.Conv2D(self.filter_outs[2]-60, kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv3_2 = tf.keras.layers.Conv2D(self.filter_outs[2]-30, kernel_size=(
                1, 7), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv3_3 = tf.keras.layers.Conv2D(self.filter_outs[2], kernel_size=(
                7, 1), padding='same', activation='elu', kernel_initializer='he_normal')

            self.conv4_1 = tf.keras.layers.Conv2D(self.filter_outs[2]-60, kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_2 = tf.keras.layers.Conv2D(self.filter_outs[2]-60, kernel_size=(
                1, 7), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_3 = tf.keras.layers.Conv2D(self.filter_outs[2]-30, kernel_size=(
                7, 1), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_4 = tf.keras.layers.Conv2D(self.filter_outs[2]-30, kernel_size=(
                1, 7), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_5 = tf.keras.layers.Conv2D(self.filter_outs[2], kernel_size=(
                7, 1), padding='same', activation='elu', kernel_initializer='he_normal')

            self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        if self.residual:
            pass
        else:
            h1 = self.avgpool(x)
            h1 = self.conv1_1(h1)

            h2 = self.conv2_1(x)

            h3 = self.conv3_1(x)
            h3 = self.conv3_2(h3)
            h3 = self.conv3_3(h3)

            h4 = self.conv4_1(x)
            h4 = self.conv4_2(h4)
            h4 = self.conv4_3(h4)
            h4 = self.conv4_4(h4)
            h4 = self.conv4_5(h4)

            return self.concat([h1, h2, h3, h4])

    def get_config(self):
        config = super().get_config()
        config.update({"filter_outs": self.filter_outs,
                       "residual": self.residual})
        return config
