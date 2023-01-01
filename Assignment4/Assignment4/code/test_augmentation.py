import os
from pathlib import Path
from unittest import TestCase

from matplotlib import pyplot as plt

from Assignment4.Assignment4.code.augmentation import Augmentor
from Assignment4.Assignment4.code.utils import load_image


class TestAugmentor(TestCase):

    def setUp(self) -> None:
        print(os.getcwd())
        path = Path(__file__).parent.parent / 'data' / 'lena.png'
        # self.image = load_image(str(path))
        # self.image = cv2.imread(str(path))
        self.image = Augmentor.vl2cv(load_image(str(path)))
        self.augmentor = Augmentor(1)

    def test_rotate_image(self):
        ret = self.augmentor.rotated_image(self.image)
        self.plt_show_cv(ret)
        plt.show()
        self.assertFalse((ret == self.image).all())

    def test_canrun(self):
        self.plt_show_cv(self.image)
        plt.show()

    @staticmethod
    def plt_show_cv(image):
        plt.imshow(image[:, :, [2, 1, 0]])

    def test_mirror(self):
        mirror = Augmentor.mirrored_image(self.image)
        self.plt_show_cv(mirror)
        plt.show()

    def test_blur(self):
        blur = Augmentor.blured_image(self.image)
        self.plt_show_cv(blur)
        plt.show()


