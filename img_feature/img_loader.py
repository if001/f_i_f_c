import keras
from keras.datasets import mnist
from PIL import Image
import numpy as np
import os
import random as rand


class ImgLoader():
    @classmethod
    def __img_open(cls, filepath):
        try:
            img = Image.open(filepath)
            img = img.convert("RGB")
            img = img.resize((28, 28))
            img = np.array(img)
        except:
            img = None
        return img

    @classmethod
    def make_train_data(cls, batch_size, img_file_dir):
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
