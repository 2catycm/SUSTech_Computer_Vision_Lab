import os
from pathlib import Path
from unittest import TestCase

import matplotlib.pyplot as plt

from utils import *


class TestUtils(TestCase):

    def setUp(self) -> None:
        print(os.getcwd())
        self.path = Path(__file__).parent.parent / 'data' / 'lena.png'

    def test_gray(self):
        img = load_image_gray(str(self.path))
        plt.imshow(img, cmap='gray')
        plt.show()
    def test_resize_gray(self):
        img = load_image_gray(str(self.path))
        print(img.shape)
        scale_factor = 0.25
        img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        print(img.shape)
        plt.imshow(img, cmap='gray')
        plt.show()
