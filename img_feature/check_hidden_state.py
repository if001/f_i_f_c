"""
学習が進行しているかを確認するため中間層を確認
"""

from char_img_autoencoder import CharImgAutoencoder
from img_loader import ImgLoader

import sys
sys.path.append("../")
from img_char.img_char_opt import ImgCharOpt

from matplotlib import pylab as plt

# グラフに日本語を表示するために必要
import matplotlib
font = {'family': 'AppleGothic'}
matplotlib.rc('font', **font)

feat_shape = 8
feat_pict_size = 4
font_size = 32


def create_feat_graph(predicted, train_yomi_list):
    fig = plt.figure("919")
    plt.subplots_adjust(hspace=0.9)

    for i, p in enumerate(predicted):
        ax = fig.add_subplot(len(predicted), 1, i + 1)
        p = p.reshape(feat_pict_size, feat_pict_size * feat_shape)
        gray = p * 255
        ax.pcolor(gray, cmap="gray")
        plt.title(str(train_yomi_list[i]))
        plt.tick_params(labelsize=5)
        plt.xlim(0, feat_shape * feat_pict_size)
    plt.savefig("./debug_data/hidden_state")
    # plt.show()


def main():
    predict_num = 10
    train_x, _ = ImgLoader.make_train_data_random(
        predict_num, "../font_img/image/hiragino/")
    img_char_opt = ImgCharOpt(
        "../font_img/image/hiragino/", "../img_char/image_save_dict/")
    train_yomi_list = []
    for t in train_x:
        t = t.reshape(font_size, font_size)
        train_yomi_list.append(img_char_opt.image2char(t))
    print(train_yomi_list)

    char_img = CharImgAutoencoder(
        "./weight/char_feature.hdf5", init_model=False)
    predicted = char_img.encoder.predict(train_x)
    create_feat_graph(predicted, train_yomi_list)


if __name__ == "__main__":
    main()
