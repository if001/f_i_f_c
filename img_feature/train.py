from char_img_autoencoder import CharImgAutoencoder
from img_loader import ImgLoader
from keras.utils import plot_model
from keras.optimizers import SGD
import numpy as np
import sys

data_size = 80000
test_size = 10000


def save_npz(save_train_file, train, teach):

    np.savez(save_train_file, train=train, teach=teach)
    print("save ", save_train_file)
    exit(0)


def load_npz(save_train_file):
    load = np.load(save_train_file)
    print("load ", save_train_file)
    train = load['train']
    teach = load['teach']

    return train, teach


def main():
    # char img autoencoderの学習

    save_train_file = "./debug_data/train_font-size64_add_font.npz"
    if sys.argv[-1] == "save":
        # train, teach = ImgLoader.make_train_data("../font_img/image/hiragino/")
        train, teach = ImgLoader.make_train_data_any_file(["../font_img/images/ricty/",
                                                           "../font_img/images/hiragino/",
                                                           "../font_img/images/hiragino_mintyou/"])
        save_npz(save_train_file, train, teach)
        exit(0)

    if len(sys.argv) == 2 and sys.argv[-1] != "save":
        weight_file = sys.argv[-1]
        # weight_file = "./weight/char_feature_simple.hdf5"
    train, teach = load_npz(save_train_file)
    print("train data:", train.shape)
    print("teach data:", teach.shape)
    # test_x, test_y = ImgLoader.make_train_data_ramdom(
    #     test_size, "../font_img/image/")
    char_img = CharImgAutoencoder(wight_file, init_model=True)

    loss = 'binary_crossentropy'
    loss = 'mean_squared_error'

    opt = 'adadelta'
    opt = 'adam'
    opt = 'sgd'
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    char_img.autoencoder.compile(optimizer=opt,
                                 loss=loss,
                                 metrics=['acc'])
    # plot_model(char_img.autoencoder, to_file='./debug_data/auto_encoder.png')

    char_img.autoencoder.fit(train, teach,
                             batch_size=128,
                             epochs=2000,
                             verbose=1,
                             validation_split=0.2,
                             # validation_data=(test_x, test_y),
                             callbacks=char_img.callback_list(log_file_name="./debug_data/training_log.csv"))
    char_img.save_model()


if __name__ == "__main__":
    main()
