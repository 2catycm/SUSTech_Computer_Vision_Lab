import cv2
import numpy as np


class Augmentor:

    def __init__(self, num_augmented_images, params=None):
        self.num_augmented_images = num_augmented_images
        # self.methods = [self.rotated_image, self.mirrored_image, self.blured_image]
        self.methods = [self.mirrored_image, self.blured_image]
        if params is None:
            self.params = {'sigma0':1.52}
        else:
            self.params = params

    def gen_augmented_image(self, cv_image):
        yield cv_image
        for i in range(self.num_augmented_images):
            yield np.random.choice(self.methods, 1)(cv_image)

    @staticmethod
    def vl2cv(vl_image):
        return vl_image.astype(np.uint8)[..., ::-1]

    def blured_image(self, image, choice=None):
        octaves = np.random.randint(0, 6)
        sigma0 = self.params.get('sigma0', 1.52)
        scale_factor = self.params.get('scale_factor', 0.65)
        sigma = sigma0 * ((1/scale_factor) ** octaves)
        return cv2.GaussianBlur(image, (0, 0), sigma0)


    # @staticmethod
    # def rotated_image(image, angle=None):
    #     if angle is None:
    #         angle = np.random.randint(-10, 10)
    #     rows, cols = image.shape[:2]
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    #     affined = cv2.warpAffine(image, M, (cols, rows))
    #     return affined

    @staticmethod
    def mirrored_image(image):
        return cv2.flip(image, 1)  # 左右翻转
