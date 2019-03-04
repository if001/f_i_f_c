from char_img_autoencoder import CharImgAutoencoder
from img_loader import ImgLoader
from matplotlib import pylab as plt

import sys
sys.path.append("../")
from img_char.img_char_opt import ImgCharOpt


def main():
    predict_num = 2
    feat_shape = 8
    pict_size = 4

    train_x, train_y = ImgLoader.make_train_data_random(
        predict_num, "../font_img/image/hiragino/")
    img_char_opt = ImgCharOpt(
        "../font_img/image/hiragino/", "../img_char/image_save_dict/")
    for t in train_x:
        t = t.reshape(32, 32)
        print(img_char_opt.image2char(t))
    exit(0)

    char_img = CharImgAutoencoder(
        "./weight/char_feature.hdf5", init_model=True)
    predicted = char_img.encoder.predict(train_x)
    print(predicted)
    fig = plt.figure()

    for i, p in enumerate(predicted):
        ax = fig.add_subplot(predict_num, 1, i + 1)
        p = p.reshape(pict_size, pict_size * feat_shape)
        gray = p * 255
        ax.pcolor(gray, cmap="gray")
        plt.tick_params(labelsize=5)
        plt.xlim(0, feat_shape * pict_size)
    plt.show()


if __name__ == "__main__":
    main()
