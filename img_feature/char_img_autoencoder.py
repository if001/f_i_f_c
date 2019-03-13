from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU, Dropout, Reshape, Flatten, Concatenate
from keras.models import Model
from keras import backend as K
import keras
from keras.callbacks import EarlyStopping, CSVLogger
import os


class CharImgAutoencoder():
    def __init__(self, save_weight_file, init_model=False):
        # self.font_size = 32
        self.font_size = 64
        self.hidden_dim = 512
        self.save_weight_file = save_weight_file
        if init_model:
            # self.autoencoder = self.__make_model()
            # self.autoencoder = self.__make_gray_scale_model_highdim()
            # self.autoencoder = self.__make_gray_scale_model_simple()
            # self.autoencoder = self.__make_gray_scale_model_deep()
            # self.autoencoder = self.__make_fully_econnected_model(debug=True)
            # self.autoencoder = self.__make_contracting_path_model(debug=True)
            # self.autoencoder = self.__make_gray_scale_flat_model()
            self.autoencoder = self.__make_some_filter_model()
            # self.encoder = self.__make_encoder_model()
            # self.decoder = self.__make_decoder_model()
        else:
            self.autoencoder = self.__load_model()
            self.encoder = self.__make_encoder_model(debug=False)
            self.decoder = self.__make_decoder_model(debug=False)

    def callback_list(self, log_file_name="./training_log.csv"):
        es_cb = EarlyStopping(
            monitor='val_loss', patience=1, verbose=1, mode='auto')
        csv_logger = CSVLogger(log_file_name)
        return [es_cb, csv_logger]

    def __make_model(self):
        input_img = Input(shape=(self.font_size, self.font_size, 3))
        x = Conv2D(16, (3, 3), padding='same')(input_img)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.6)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.6)(x)
        x = Conv2D(8, (3, 3),  padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.6)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3),  padding='same')(x)
        x = ReLU()(x)
        encoded = MaxPooling2D((2, 2), padding='same', name="encoder")(x)
        # encoder = Model(input_img, encoded)

        x = Conv2D(8, (3, 3),  padding='same', name="decoder")(encoded)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.6)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3),  padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.6)(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.6)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3))(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.6)(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder

    def __up_block(self, inp, unit, name=None):
        x = Conv2D(unit, (3, 3), padding='same', name=name)(inp)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        return x

    def __down_block(self, inp, unit, name=None):
        x = Conv2D(unit, (3, 3), padding='same', name=name)(inp)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)
        return x

    def __conv_block(self, inp, unit, kernel_size, strides, name_first=None, name_last=None):
        x = Conv2D(unit, kernel_size=kernel_size, strides=strides,
                   padding='same', name=name_first)(inp)
        x = BatchNormalization()(x)
        x = ReLU(name=name_last)(x)
        return x

    def __make_some_filter_model(self):
        input_img = Input(shape=(self.font_size, self.font_size, 1))
        x1 = self.__conv_block(input_img, 48, (2, 2), (2, 2))
        x1 = MaxPooling2D((2, 2), padding='same')(x1)
        x2 = self.__conv_block(input_img, 48, (2, 5), (2, 2))
        x2 = MaxPooling2D((2, 2), padding='same')(x2)
        x3 = self.__conv_block(input_img, 48, (5, 2), (2, 2))
        x3 = MaxPooling2D((2, 2), padding='same')(x3)
        x4 = self.__conv_block(input_img, 48, (5, 5), (2, 2))
        x4 = MaxPooling2D((2, 2), padding='same')(x4)
        x5 = self.__conv_block(input_img, 48, (10, 10), (2, 2))
        x5 = MaxPooling2D((2, 2), padding='same')(x5)
        x6 = self.__conv_block(input_img, 48, (5, 10), (2, 2))
        x6 = MaxPooling2D((2, 2), padding='same')(x6)
        x7 = self.__conv_block(input_img, 48, (10, 5), (2, 2))
        x7 = MaxPooling2D((2, 2), padding='same')(x7)

        x = Concatenate()([x1, x2, x3, x4, x5, x6, x7])
        x = self.__conv_block(x, 256, (3, 3), (1, 1))
        x = Concatenate()([x1, x2, x3, x4, x5])
        x = self.__conv_block(x, 128, (3, 3), (1, 1))
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = self.__conv_block(x, 64, (3, 3), (1, 1))
        x = MaxPooling2D((2, 2), padding='same', name="encoder")(x)

        x = self.__conv_block(x, 64, (3, 3), (1, 1), name_first="decoder")
        x = UpSampling2D((2, 2))(x)
        x = self.__conv_block(x, 128, (3, 3), (1, 1))
        x = UpSampling2D((2, 2))(x)
        x = self.__conv_block(x, 256, (3, 3), (1, 1))
        x = UpSampling2D((2, 2))(x)
        x = self.__conv_block(x, 256, (3, 3), (1, 1))
        x = UpSampling2D((2, 2))(x)
        decoded = self.__conv_block(x, 1, (3, 3), (1, 1))
        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder

    def __make_gray_scale_flat_model(self):
        input_img = Input(shape=(self.font_size, self.font_size, 1))
        x = self.__conv_block(input_img, 48, (5, 5), (2, 2))
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = self.__conv_block(x, 64, (3, 3), (1, 1))
        x = self.__conv_block(x, 128, (3, 3), (1, 1))
        # ------------------------
        x = self.__conv_block(x, 256, (3, 3), (2, 2))
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = self.__conv_block(x, 256, (3, 3), (1, 1))
        x = self.__conv_block(x, 256, (3, 3), (1, 1))
        # ------------------------
        x = self.__conv_block(x, 256, (3, 3), (2, 2))
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = self.__conv_block(x, 256, (3, 3), (1, 1))

        x = self.__conv_block(x, 512, (3, 3), (1, 1))
        x = self.__conv_block(x, 512, (3, 3), (1, 1))
        x = self.__conv_block(x, 512, (3, 3), (1, 1), name_last="encoder")
        x = self.__conv_block(x, 512, (3, 3), (1, 1), name_first="decoder")
        x = self.__conv_block(x, 512, (3, 3), (1, 1))
        x = self.__conv_block(x, 512, (3, 3), (1, 1))

        x = self.__conv_block(x, 256, (3, 3), (1, 1))
        x = self.__conv_block(x, 64, (3, 3), (1, 1))
        # ------------------------
        x = UpSampling2D((2, 2))(x)
        x = self.__conv_block(x, 64, (3, 3), (1, 1))
        x = UpSampling2D((2, 2))(x)
        x = self.__conv_block(x, 64, (3, 3), (1, 1))
        # ------------------------
        x = UpSampling2D((2, 2))(x)
        x = self.__conv_block(x, 64, (3, 3), (1, 1))
        x = UpSampling2D((2, 2))(x)
        x = self.__conv_block(x, 48, (3, 3), (1, 1))
        # ------------------------
        x = UpSampling2D((2, 2))(x)
        x = self.__conv_block(x, 48, (3, 3), (1, 1))
        x = UpSampling2D((2, 2))(x)
        x = self.__conv_block(x, 24, (3, 3), (1, 1))
        decoded = Conv2D(1, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', activation="sigmoid")(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder

    def __make_gray_scale_model_highdim(self):
        input_img = Input(shape=(self.font_size, self.font_size, 1))

        x = self.__up_block(input_img, 16)
        x = self.__up_block(x, 32)
        x = self.__up_block(x, 64)
        x = self.__up_block(x, 128)
        x = self.__up_block(x, 256)
        x = Conv2D(self.hidden_dim, (3, 3), padding='same')(x)
        x = ReLU()(x)
        encoded = MaxPooling2D((2, 2), padding='same', name="encoder")(x)

        x = self.__down_block(encoded, 256, name="decoder")
        x = self.__down_block(x, 128)
        x = self.__down_block(x, 64)
        x = self.__down_block(x, 64)
        x = self.__down_block(x, 32)
        x = self.__down_block(x, 16)

        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder

    def __make_gray_scale_model_simple(self):
        input_img = Input(shape=(self.font_size, self.font_size, 1))

        x = Conv2D(16, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.hidden_dim, (3, 3),
                   padding='same', activation="sigmoid")(x)

        encoded = MaxPooling2D((2, 2), padding='same', name="encoder")(x)

        x = Conv2D(8, (3, 3), padding='same', name="decoder")(encoded)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(8, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)

        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder

    def __make_gray_scale_model_deep(self):
        input_img = Input(shape=(self.font_size, self.font_size, 1))

        x = Conv2D(16, (3, 3), padding='same')(input_img)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(self.hidden_dim, (3, 3), padding='same')(x)
        x = ReLU()(x)
        encoded = MaxPooling2D((2, 2), padding='same', name="encoder")(x)

        x = UpSampling2D((2, 2), name="decoder")(encoded)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = UpSampling2D((2, 2))(x)

        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder

    def __make_fully_econnected_model(self, debug=True):
        input_img = Input(shape=(self.font_size, self.font_size, 1))

        x = Conv2D(16, (3, 3), padding='same')(input_img)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Dropout(0.7)(x)
        x = Flatten()(x)
        x = Dense(8 * 8 * 8, activation="relu")(x)
        x = Dropout(0.7)(x)
        x = Dense(4 * 4 * 8, activation="relu")(x)
        x = Dropout(0.7)(x)
        encoded = Dense(4 * 4 * 8, activation="sigmoid",
                        name="encoder")(x)

        x = Dense(8 * 8 * 8, name="decoder")(encoded)
        x = Dropout(0.7)(x)
        x = Dense(16 * 16 * 16, activation="relu")(x)
        x = Dropout(0.7)(x)
        x = Reshape((16, 16, 16))(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder

    def residual_block(self, inp, unit, name=None):
        x = Conv2D(unit, (3, 3), padding='same', name=name)(inp)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Concatenate()([inp, x])
        x = Conv2D(unit, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def __make_contracting_path_model(self, debug=True):
        input_img = Input(shape=(self.font_size, self.font_size, 1))

        x = self.residual_block(input_img, 16)
        x = MaxPooling2D((4, 4))(x)
        x = self.residual_block(x, 16)
        x = MaxPooling2D((4, 4), name="encoder")(x)

        x = self.residual_block(x, 16, name="decoder")
        x = UpSampling2D((4, 4))(x)
        x = self.residual_block(x, 16)
        x = UpSampling2D((4, 4))(x)
        decoded = Conv2D(1, (3, 3), padding='same', activation="sigmoid")(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder

    def __make_encoder_contractiong_path_model(self):
        encoder_layer = self.__search_layer("encoder")
        idx = self.autoencoder.layers.index(encoder_layer)
        l = self.autoencoder.layers[1:idx + 1]

        inp = Input(shape=(self.font_size, self.font_size, 1))
        x = l[1](inp)
        x = l[2](x)
        x = l[3](x)
        x = Concatenate()([inp, x])

        x = l[5](x)
        x = l[6](x)
        x = l[7](x)
        c = l[8](x)
        x = l[9](c)
        x = l[10](x)
        x = l[11](x)
        x = Concatenate()([c, x])

        x = l[13](x)
        x = l[14](x)
        x = l[15](x)
        x = l[16](x)
        encoder_model = Model(inp, x)

    def __make_decoder_contractiong_path_model(self):
        decoder_layer = self.__search_layer("decoder")
        idx = self.autoencoder.layers.index(decoder_layer)
        l = self.autoencoder.layers[idx:]

        inp = Input(shape=(2, 2, self.hidden_dim))
        x = l[0](inp)
        x = l[1](x)
        x = l[2](x)
        x = Concatenate()([inp, x])

        x = l[4](x)
        x = l[5](x)
        x = l[6](x)
        x = l[7](x)
        c = l[8](x)
        x = l[9](c)
        x = l[10](x)
        x = Concatenate()([c, x])

        x = l[13](x)
        x = l[14](x)
        x = l[15](x)
        x = l[16](x)
        encoder_model = Model(inp, x)

    def __make_encoder_model(self, debug=True):
        l = self.__search_layer("encoder")
        idx = self.autoencoder.layers.index(l)
        inp = Input(shape=(self.font_size, self.font_size, 1))
        # inp = Input(shape=(28, 28,3))# color scale
        x = inp
        for encoder in self.autoencoder.layers[1:idx + 1]:
            x = encoder(x)
        encoder_model = Model(inp, x)
        if debug:
            encoder_model.summary()
        return encoder_model

    def __make_decoder_model(self, debug=True):
        l = self.__search_layer("decoder")
        idx = self.autoencoder.layers.index(l)
        inp = Input(shape=(1, 1, self.hidden_dim))
        # inp = Input(shape=(2, 2, self.hidden_dim))
        # inp = Input(shape=(4 * 4 * 8,))
        x = inp
        for decoder in self.autoencoder.layers[idx:]:
            x = decoder(x)
        decoder_model = Model(inp, x)
        if debug:
            decoder_model.summary()
        return decoder_model

    def __search_layer(self, name):
        res = None
        for l in self.autoencoder.layers:
            if l.name == name:
                res = l
        return res

    def save_model(self):
        self.autoencoder.save(self.save_weight_file)
        print("save " + self.save_weight_file)

    def __load_model(self):
        model = load_model(self.save_weight_file)
        print("load " + self.save_weight_file)
        from keras.models import load_model
        return model
