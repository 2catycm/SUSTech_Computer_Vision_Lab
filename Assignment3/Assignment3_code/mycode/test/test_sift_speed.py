#%%
import utils
p = "/mnt/e/AsciiStandardPath/PracticeFile/22fall/P_Computer_Vision/SUSTech_Computer_Vision_Lab/Assignment3/Assignment3_code/data/train/Bedroom/image_0001.jpg"
image = utils.load_image_gray(p)
image.shape
from scipy import stats
stats.describe(image.flatten())
# %%
import cyvlfeat as vlfeat

%timeit frames, descriptors = vlfeat.sift.dsift(image, step=1, fast=True)
# %%
import cv2
sift = cv2.SIFT_create()
cv_image = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX).astype('uint8')
%timeit kp, des = sift.detectAndCompute(cv_image,None)


# %%
import torch as np
np.array = lambda x: np.Tensor(x).to('cuda')
jt_image = np.array(cv_image)

#%%
import cyvlfeat as vlfeat
vlf_image_gpu = np.array(image)

frames, descriptors = vlfeat.sift.dsift(vlf_image_gpu, step=5, fast=True)
# %%
%timeit kp, des = sift.detectAndCompute(jt_image,None)

# %%
import PythonSIFT.pysift as pysift

keypoints, descriptors = pysift.computeKeypointsAndDescriptors(jt_image)

# %%
from numba import cuda
@cuda.jit
def test():
    print("Hello World")
test()

# %%
