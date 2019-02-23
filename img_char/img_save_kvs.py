"""
画像のファイル名と読み方(char)を対応させるkvs
"""

import plyvel
import numpy as np
from itertools import chain
import json
from PIL import Image
import os
import sys


class ImageSaveKvs():
    def __init__(self, save_db_path="./image_save_dict/", save_db_name="image.ldb"):
        db_path = os.path.join(save_db_path, save_db_name)
        self.my_db = plyvel.DB(db_path, create_if_missing=True)

    def __del__(self):
        self.my_db.close()

    def put(self, key, value):
        """
        key: type:string
        value: type: list object
        """
        self.my_db.put(self.__u(key), self.__b(value))

    def get(self, key):
        """
        return int array
        """
        value = self.my_db.get(self.__u(key))

        if value is not None:
            value = [x for x in value]
        return value

    def delete(self, key):
        self.my_db.delete(self.__u(key))

    def __u(self, st):
        return st.encode('utf-8')

    def __b(self, byte_obj):
        return bytes(byte_obj)


def main():
    i2k = ImageSaveKvs()
    print(i2k.get("21"))


if __name__ == "__main__":
    main()
x
