from PIL import Image
import plyvel
import numpy as np
import os
import sys
sys.path.append("../")
from img_char.img_save_kvs import ImageSaveKvs


class ImgCharOpt():
    def __init__(self, image_save_path="../font_img/image", db_path="./image_save_dict/"):
        self.image_save_path = image_save_path
        self.db_path = db_path

    def exclude_extension(self, st):
        return st.split(".")[0]

    def similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def extension(self, st):
        return st.split(".")[1]

    def load_image(self, yomi):
        img = Image.open(self.image_save_path + yomi + "_0.png")
        img = img.convert("RGB")
        img = img.resize((28, 28))
        img = np.array(img)
        return img

    def image2char(self, img):
        dig_sim = 0
        label = ""
        img = img.flatten()
        files = os.listdir(self.image_save_path)
        i2k = ImageSaveKvs(save_db_path=self.db_path)

        for fname in files:
            if self.extension(fname) == "png" and (self.exclude_extension(fname).split("_")[1] == "0"):
                yomi = self.exclude_extension(fname).split("_")[0]
                load_img = i2k.get(yomi)
                sim = self.similarity(img, load_img)
                if dig_sim < sim:
                    dig_sim = sim
                    char = bytes.fromhex(yomi).decode('utf-8')
        print("similarity:", dig_sim)
        return char

    def char2image(self, char):
        bytes_yomi = char.encode("UTF-8").hex()
        image_path = os.path.join(self.image_save_path, bytes_yomi + "_.png")
        return self.load_image(image_path)


def all_save_kvs():
    image_file_dir = "../font_img/image/"
    image_files = os.listdir(image_file_dir)

    i2k = ImageSaveKvs()
    img_char_opt = ImgCharOpt()
    for fname in image_files:
        if "_0" in img_char_opt.exclude_extension(fname):
            yomi = img_char_opt.exclude_extension(fname).split("_")[0]
            print(yomi)
            img = img_char_opt.load_image(yomi)
            img = img.flatten()
            if i2k.get(yomi) is None:
                i2k.put(yomi, img)


def main():
    all_save_kvs()


if __name__ == "__main__":
    main()
