import cv2
import numpy as np


class Augmentor:

    def __init__(self, num_augmented_images):
        self.num_augmented_images = num_augmented_images
        self.methods = [self.rotated_image, self.mirrored_image, self.blured_image]

    def gen_augmented_image(self, cv_image):
        yield cv_image
        method = np.random.choice(self.methods, self.num_augmented_images)
        for m in method:
            yield m(cv_image)

    @staticmethod
    def vl2cv(vl_image):
        return vl_image.astype(np.uint8)[..., ::-1]

    @staticmethod
    def blured_image(image, choice=None):
        if choice is None:
            choice = np.random.randint(0, 3)
        if choice == 0:
            ksize = np.random.randint(3, 7)
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif choice == 1:
            ksize = np.random.randint(3, 7)
            return cv2.MediaBlur(image, (ksize, ksize))
        elif choice == 2:
            return cv2.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def rotated_image(image, angle=None):
        if angle is None:
            angle = np.random.randint(-10, 10)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        affined = cv2.warpAffine(image, M, (cols, rows))
        return affined

    @staticmethod
    def mirrored_image(image):
        return cv2.flip(image, 1)  # 左右翻转
