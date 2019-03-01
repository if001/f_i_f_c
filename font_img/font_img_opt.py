import os
import unicodedata
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
import plyvel
import numpy as np
import os
import sys


class FontImgOpt():
    def __init__(self, image_save_path="./image/", font_path="./fonts/"):
        self.font_size = 32
        self.font_size_en = 36
        self.pict_height = 32
        self.pict_width = 32

        self.font_file = os.path.join(
            font_path, 'RictyDiminished-Regular.ttf')
        self.font_file = os.path.join(
            font_path, 'hiragino_maru_go_ProN_W4.ttc')
        self.img_save_dir = os.path.join(image_save_path)

    def save_image(self, yomi, font, prefix, processing=None, pos=(0, 0), rad=None):
        image = Image.new(
            'RGB', (self.pict_height, self.pict_width), (255, 255, 255))

        draw = ImageDraw.Draw(image)
        draw.text(pos, yomi, font=font, fill='#000000')

        if processing is not None:
            image = processing(image)
        if rad is not None:
            image = image.rotate(rad)
        print("save " + self.img_save_dir + yomi + "_" + prefix + ".png")
        bytes_yomi = yomi.encode("UTF-8").hex()
        print(bytes_yomi)
        image.save(self.img_save_dir + bytes_yomi +
                   "_" + "test" + prefix + ".png", 'PNG')

    def is_exist(self, yomi):
        return os.path.isfile(self.img_save_dir + yomi + '.png')

    def char2font(self, char, font_file=None):
        yomi_str = char
        if font_file is None:
            font_file = self.font_file
        print(font_file)
        if unicodedata.east_asian_width(char) in 'FWA':  # 全角のとき
            font = ImageFont.truetype(
                font_file, self.font_size, encoding='unic')
        else:
            font = ImageFont.truetype(
                font_file, self.font_size_en, encoding='unic')
        return font

    def create_font_img(self, yomi_str, font, font_file=None):
        img_pos = [(0, 0), (0, 2), (0, 3), (2, 0), (3, 0), (2, 2), (3, 3)]
        for i in range(len(img_pos)):
            prefix = str(i)
            if not self.is_exist(yomi_str + prefix):
                self.save_image(yomi_str, font, prefix, pos=img_pos[i])

        shape_method_set = [
            # ShapeMethodSet("flip", func=ImageOps.flip),
            # ShapeMethodSet("mirror", func=ImageOps.mirror),
            # ShapeMethodSet("rotate_90", rad=90),
            # ShapeMethodSet("rotate_180", rad=180),
            ShapeMethodSet("contrast_50", func=ShapeMethod.contrast_50),
            ShapeMethodSet("sharpness_0", func=ShapeMethod.sharpness_0),
            ShapeMethodSet("sharpness_2", func=ShapeMethod.sharpness_2),
            # ShapeMethodSet("gaussianblur", func=ShapeMethod.gaussian_bluer),
            ShapeMethodSet("erosion", func=ShapeMethod.erosion),
            # ShapeMethodSet("dilation", func=ShapeMethod.dilation),
        ]

        for shape_method in shape_method_set:
            if not self.is_exist(yomi_str + "_" + shape_method.name):
                self.save_image(yomi_str, font,
                                shape_method.name,
                                processing=shape_method.func,
                                rad=shape_method.rad)
        exit(0)


class ShapeMethodSet():
    def __init__(self, name, func=None, rad=None):
        self.name = name
        self.func = func
        self.rad = rad


class ShapeMethod():
    @classmethod
    def contrast_50(cls, img):
        img = ImageEnhance.Contrast(img)
        img = img.enhance(0.5)
        return img

    @classmethod
    def sharpness_2(cls, img):
        img = ImageEnhance.Sharpness(img)
        img = img.enhance(2.0)  # シャープ画像
        return img

    @classmethod
    def sharpness_0(cls, img):
        img = ImageEnhance.Sharpness(img)
        img = img.enhance(0.0)  # ボケ画像
        return img

    @classmethod
    def gaussian_bluer(cls, img):
        img = img.filter(ImageFilter.GaussianBlur(3.0))
        return img

    @classmethod
    def erosion(cls, img):  # 縮小
        img = img.filter(ImageFilter.MinFilter())
        return img

    @classmethod
    def dilation(cls, img):  # 膨張
        img = img.filter(ImageFilter.MaxFilter())
        return img


def main():
    with open("../data/files_all_rnp.txt") as f:
        char_list = list(set(list(f.read())))

    if '\n' in char_list:
        char_list.remove('\n')
    if ' ' in char_list:
        char_list.remove(' ')
    if '' in char_list:
        char_list.remove('')

    font_img_opt = FontImgOpt()
    for char in char_list:
        font = font_img_opt.char2font(char)
        font_img_opt.create_font_img(char, font)


if __name__ == '__main__':
    main()
