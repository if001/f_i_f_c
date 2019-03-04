from PIL import Image
import plyvel
import numpy as np
import os
import sys
sys.path.append("../")
from img_char.img_save_kvs import ImageSaveKvs


class ImgCharOpt():
    def __init__(self, image_save_path="../font_img/image/", db_path="./image_save_dict/"):
        self.image_save_path = image_save_path
        self.i2k = ImageSaveKvs(save_db_path=db_path)
        self.font_size = 32

    def exclude_extension(self, st):
        return st.split(".")[0]

    def similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def extension(self, st):
        return st.split(".")[1]

    def load_image(self, yomi):
        img = Image.open(self.image_save_path + yomi + "_0.png")
        # img = img.convert("RGB")
        img = img.convert("L")
        img = img.resize((self.font_size, self.font_size))
        img = np.array(img)
        return img

    def image2char(self, img):
        """
        (28,28,3)
        """
        dig_sim = 0
        label = ""
        img = img.flatten()
        files = os.listdir(self.image_save_path)

        for fname in files:
            if (self.extension(fname)) == "png" and (self.exclude_extension(fname).split("_")[1] == "0"):
                yomi = self.exclude_extension(fname).split("_")[0]
                load_img = self.i2k.get(yomi)
                print(img.shape)
                load_img = np.array(load_img)
                print(load_img.shape)
                exit(0)
                sim = self.similarity(img, load_img)
                if dig_sim < sim:
                    dig_sim = sim
                    char = bytes.fromhex(yomi).decode('utf-8')
        print("similarity:", dig_sim)
        return char

    def char2image(self, char):
        bytes_yomi = char.encode("UTF-8").hex()
        img = self.i2k.get(bytes_yomi)
        img = np.array(img)
        # img.resize((self.font_size, self.font_size, 3)) #color
        img.resize((self.font_size, self.font_size, 1))  # gray
        return img
        # image_path = os.path.join(self.image_save_path, bytes_yomi + "_0.png")
        # return self.load_image(image_path)


def all_save_kvs():
    """
    フォント画像は複数のディレクトリに別れているが、
    読みと保存した画像ファイル名はディレクトリごとに同じなので
    1つのディレクトリに保存されてる画像ファイル名のみマッピングさせる
    """
    image_file_dir = "../font_img/image/hiragino/"
    image_files = os.listdir(image_file_dir)

    img_char_opt = ImgCharOpt(image_file_dir)
    for fname in image_files:
        if "_0" in img_char_opt.exclude_extension(fname):
            yomi = img_char_opt.exclude_extension(fname).split("_")[0]
            print(yomi)
            img = img_char_opt.load_image(yomi)
            img = img.flatten()

            if img_char_opt.i2k.get(yomi) is None:
                img_char_opt.i2k.put(yomi, img)


def main():
    all_save_kvs()


if __name__ == "__main__":
    main()
