import sys
sys.path.append("../")
from img_feature.char_img_autoencoder import CharImgAutoencoder
import numpy as np


class ImgFeatureOpt():
    def __init__(self, char_auto_encoder_weight_file="./weight/char_feature.hdf5"):
        self.char_img = CharImgAutoencoder(char_auto_encoder_weight_file)

    def img2feature(self, img):
        """
        input_dim: (28,28,3) color
        input_dim: (28,28,1)
        input: image color range 0~255
        """
        img = img / 255.
        result = self.char_img.encoder.predict(np.array([img]))
        return np.array(result[0])

    def feature2img(self, feature):
        """
        input_dim: (4,4,8)
        """
        result = self.char_img.decoder.predict(np.array([feature]))
        result = np.array(result[0]) * 255.
        return result


def main():
    pass


if __name__ == "__main__":
    main()
