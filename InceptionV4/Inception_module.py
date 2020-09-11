import tensorflow as tf

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


class Inception_C(tf.keras.layers.Layer):
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

            self.conv3_1 = tf.keras.layers.Conv2D(self.filter_outs[2]+128, kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv3_2_1 = tf.keras.layers.Conv2D(self.filter_outs[2], kernel_size=(
                1, 3), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv3_2_2 = tf.keras.layers.Conv2D(self.filter_outs[2], kernel_size=(
                3, 1), padding='same', activation='elu', kernel_initializer='he_normal')

            self.conv4_1 = tf.keras.layers.Conv2D(self.filter_outs[3]+128, kernel_size=(
                1, 1), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_2 = tf.keras.layers.Conv2D(self.filter_outs[3]+192, kernel_size=(
                1, 3), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_3 = tf.keras.layers.Conv2D(self.filter_outs[3]+256, kernel_size=(
                3, 1), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_4_1 = tf.keras.layers.Conv2D(self.filter_outs[3], kernel_size=(
                3, 1), padding='same', activation='elu', kernel_initializer='he_normal')
            self.conv4_4_2 = tf.keras.layers.Conv2D(self.filter_outs[3], kernel_size=(
                1, 3), padding='same', activation='elu', kernel_initializer='he_normal')

            self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        if self.residual:
            pass

        else:
            h1 = self.avgpool(x)
            h1 = self.conv1_1(h1)

            h2 = self.conv2_1(x)

            h3 = self.conv3_1(x)
            h3_1 = self.conv3_2_1(h3)
            h3_2 = self.conv3_2_2(h3)

            h4 = self.conv4_1(x)
            h4 = self.conv4_2(h4)
            h4 = self.conv4_2(h4)
            h4_1 = self.conv4_4_1(h4)
            h4_2 = self.conv4_4_2(h4)

            return self.concat([h1, h2, h3_1, h3_2, h4_1, h4_2])

    def get_config(self):
        config = super().get_config()
        config.update({"filter_outs": self.filter_outs,
                       "residual": self.residual})
        return config
