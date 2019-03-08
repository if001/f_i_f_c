import keras
from keras.datasets import mnist
from PIL import Image
import numpy as np
import os
import random as rand


class ImgLoader():
    @classmethod
    def __img_open(cls, filepath):
        # font_size = 32
        font_size = 64
        try:
            img = Image.open(filepath)
            # img = img.convert("RGB")
            img = img.convert('L')
            img = img.resize((font_size, font_size))
            img = np.array(img)
            img = img.reshape(font_size, font_size, 1)
        except:
            img = None
        return img

    def __create_img_from_nparr(cls, img_name, np_array):
        img_arr = Image.fromarray(np.uint8(np_array))
        img_arr.save(img_name)

    @classmethod
    def make_train_data(cls, img_file_dir):
        image_list = []
        files = os.listdir(img_file_dir)

        for f in files:
            if (f.split(".")[-1] == "png"):
                filepath = os.path.join(img_file_dir, f)
                img = cls.__img_open(filepath)
                image_list.append(img / 255.)

        image_list = np.array(image_list)
        return image_list, image_list

    @classmethod
    def make_train_data_any_file(cls, img_file_dir_list):
        image_list = []
        for img_file_dir in img_file_dir_list:
            print("start :", img_file_dir)
            files = os.listdir(img_file_dir)
            for f in files:
                if (f.split(".")[-1] == "png"):
                    filepath = os.path.join(img_file_dir, f)
                    img = cls.__img_open(filepath)
                    image_list.append(img / 255.)

        image_list = np.array(image_list)
        return image_list, image_list

    @classmethod
    def make_train_data_random(cls, batch_size, img_file_dir):
        image_list = []
        files = os.listdir(img_file_dir)

        while(True):
            idx = rand.randint(0, len(files) - 1)
            if (files[idx].split(".")[-1] == "png"):
                filepath = os.path.join(img_file_dir, files[idx])
                img = cls.__img_open(filepath)
                image_list.append(img / 255.)
            if len(image_list) == batch_size:
                break

        image_list = np.array(image_list)
        return image_list, image_list

    @classmethod
    def make_predict_data(cls, img_file_dir):
        image_label_list = []

        for fname in os.listdir(img_file_dir):
            if((fname.split(".")[-1] == "png")
                    and ("_0" in fname)):
                filepath = os.path.join(img_file_dir, fname)
                img = cls.__img_open(filepath)
                image_label_list.append([fname.split(".")[0], img / 255.])

        return image_label_list

    @classmethod
    def make_test_data(cls):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
