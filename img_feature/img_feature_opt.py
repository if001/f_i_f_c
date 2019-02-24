sys.path.append("../")
from img_feature.char_img_autoencoder import CharImgAutoencoder


class ImgFeatureOpt():
    def __init__(self, char_auto_encoder_weight_file="./weight/char_feature.hdf5"):
        self.char_img = CharImgAutoencoder(char_auto_encoder_weight_file)

    def img2feature(self, img):
        result = self.char_img.encoder(img)
        return result[0]

    def feature2img(self, feature):
        result = self.char_img.decoder(feature)
        return result[0]


def main():
    pass


if __name__ == "__main__":
    main()
