from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
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
        else:
            self.autoencoder = self.__load_model()
        self.encoder = self.__make_encoder_model()
        self.decoder = self.__make_encoder_model()

    def callback_list(self):
        es_cb = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, verbose=1, mode='auto')
        return [es_cb]

    def __make_model(self):
        input_img = Input(shape=(28, 28, 3))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        # encoder = Model(input_img, encoded)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam',
                            loss='mean_squared_error')
        autoencoder.summary()
        return autoencoder

    def __make_encoder_model(self):
        encode = self.autoencoder.layers[0:7]

        input_img = Input(shape=(28, 28, 3))
        x = encode[1](input_img)
        x = encode[2](x)
        x = encode[3](x)
        x = encode[4](x)
        x = encode[5](x)
        encoded = encode[6](x)

        encoder = Model(input_img, encoded)
        encoder.summary()
        return encoder

    def __make_decoder_model(self):
        decode = self.autoencoder.layers[7:]
        input_img = Input(shape=(4, 4, 8))
        x = decode[0](input_img)
        x = decode[1](x)
        x = decode[2](x)
        x = decode[3](x)
        x = decode[4](x)
        x = decode[5](x)
        decoded = decode[6](x)

        decoder = Model(input_img, decoded)
        decoder.summary()
        return decoder

    def save_model(self):
        self.autoencoder.save(self.save_weight_file)
        print("save " + self.save_weight_file)

    def __load_model(self):
        print("load " + self.save_weight_file)
        from keras.models import load_model
        return load_model(self.save_weight_file)
