"""
decodeした画像を確認
"""

from char_img_autoencoder import CharImgAutoencoder
from img_loader import ImgLoader

import sys
sys.path.append("../")
from img_char.img_char_opt import ImgCharOpt

from matplotlib import pylab as plt
from PIL import Image
import numpy as np


# グラフに日本語を表示するために必要
import matplotlib
font = {'family': 'AppleGothic'}
matplotlib.rc('font', **font)

feat_shape = 8
feat_pict_size = 4
font_size = 32


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def toPILarray(np_arr):
    np_arr = np_arr * 255
    pilImg = Image.fromarray(np.uint8(np_arr))
    return pilImg


def main():
    predict_num = 10
    train_x, _ = ImgLoader.make_train_data_random(
        predict_num, "../font_img/image/hiragino/")
    img_char_opt = ImgCharOpt(
        "../font_img/image/hiragino/", "../img_char/image_save_dict/")
    # train_yomi_list = []
    # for t in train_x:
    #     t = t.reshape(font_size, font_size)
    #     train_yomi_list.append(img_char_opt.image2char(t))
    # print(train_yomi_list)

    char_img = CharImgAutoencoder(
        "./weight/char_feature.hdf5", init_model=False)
    predicted = char_img.autoencoder.predict(train_x)

    compare_img_list = []
    for train, predict in zip(train_x, predicted):
        train = train.reshape(font_size, font_size)
        predict = predict.reshape(font_size, font_size)
        train = toPILarray(train)
        predict = toPILarray(predict)

        compare_img_list.append(get_concat_h(train, predict))

    concat = compare_img_list[0]
    for i in range(1, len(compare_img_list)):
        concat = get_concat_v(compare_img_list[i], concat)
    concat.save("./debug_data/test.png")


if __name__ == "__main__":
    main()
