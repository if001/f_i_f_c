from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU
from keras.models import Model
from keras import backend as K
import keras
from keras import callbacks
import os


class CharImgAutoencoder():
    def __init__(self, save_weight_file, init_model=False):
        self.save_weight_file = save_weight_file
        if init_model:
            self.autoencoder = self.__make_model()
            self.encoder = self.__make_encoder_model()
            self.decoder = self.__make_decoder_model()
        else:
            self.autoencoder = self.__load_model()
            self.encoder = self.__make_encoder_model(debug=False)
            self.decoder = self.__make_decoder_model(debug=False)

    def callback_list(self):
        es_cb = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, verbose=1, mode='auto')
        return [es_cb]

    def __make_model(self):
        input_img = Input(shape=(28, 28, 3))
        x = Conv2D(16, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(8, (3, 3),  padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3),  padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same', name="encoder")(x)
        # encoder = Model(input_img, encoded)

        x = Conv2D(8, (3, 3),  padding='same', name="decoder")(encoded)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3),  padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        # loss = 'mean_squared_error'
        # loss = 'binary_crossentropy'
        # autoencoder.compile(optimizer='adam',
        #                     loss=loss)
        autoencoder.summary()
        return autoencoder

    def __make_encoder_model(self, debug=True):
        l = self.__search_layer("encoder")
        idx = self.autoencoder.layers.index(l)

        inp = Input(shape=(28, 28, 3))
        x = inp
        for encoder in self.autoencoder.layers[:idx + 1]:
            x = encoder(x)
        encoder_model = Model(inp, x)
        if debug:
            encoder_model.summary()
        return encoder_model

        # encode = self.autoencoder.layers[0:7]

        # input_img = Input(shape=(28, 28, 3))
        # x = encode[1](input_img)
        # x = encode[2](x)
        # x = encode[3](x)
        # x = encode[4](x)
        # x = encode[5](x)
        # encoded = encode[6](x)

        # encoder = Model(input_img, encoded)
        # if debug:
        #     encoder.summary()
        # return encoder

    def __make_decoder_model(self, debug=True):
        l = self.__search_layer("decoder")
        idx = self.autoencoder.layers.index(l)

        inp = Input(shape=(4, 4, 8))
        x = inp
        for decoder in self.autoencoder.layers[idx:]:
            x = decoder(x)
        decoder_model = Model(inp, x)
        if debug:
            decoder_model.summary()
        return decoder_model

        # decode = self.autoencoder.layers[7:]
        # input_img = Input(shape=(4, 4, 8))
        # x = decode[0](input_img)
        # x = decode[1](x)
        # x = decode[2](x)
        # x = decode[3](x)
        # x = decode[4](x)
        # x = decode[5](x)
        # decoded = decode[6](x)

        # decoder = Model(input_img, decoded)
        # if debug:
        #     decoder.summary()
        # return decoder

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
        print("load " + self.save_weight_file)
        from keras.models import load_model
        return load_model(self.save_weight_file)
