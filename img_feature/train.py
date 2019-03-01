from char_img_autoencoder import CharImgAutoencoder
from img_loader import ImgLoader

data_size = 80000
test_size = 10000


def main():
    # char img autoencoderの学習
    # train_x, train_y = ImgLoader.make_train_data("../font_img/image/")
    train_x, train_y = ImgLoader.make_train_data_two_file("../font_img/image/ricty/",
                                                          "../font_img/image/hiragino/",
                                                          )

    # test_x, test_y = ImgLoader.make_train_data_ramdom(
    #     test_size, "../font_img/image/")
    char_img = CharImgAutoencoder(
        "./weight/char_feature.hdf5", init_model=True)

    loss = 'binary_crossentropy'
    loss = 'mean_squared_error'

    opt = 'adadelta'
    opt = 'adam'
    char_img.autoencoder.compile(optimizer=opt,
                                 loss=loss,
                                 metrics=['acc'])

    char_img.autoencoder.fit(train_x, train_y,
                             batch_size=128,
                             epochs=2000,
                             verbose=1,
                             validation_split=0.2,
                             # validation_data=(test_x, test_y),
                             callbacks=char_img.callback_list())
    char_img.save_model()


if __name__ == "__main__":
    main()
