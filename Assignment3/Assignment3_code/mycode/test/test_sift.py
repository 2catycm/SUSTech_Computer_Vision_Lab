#%%
import utils

p = "/mnt/e/AsciiStandardPath/PracticeFile/22fall/P_Computer_Vision/SUSTech_Computer_Vision_Lab/Assignment3/Assignment3_code/data/train/Bedroom/image_0001.jpg"
image = utils.load_image_gray(p)
#%%
import cyvlfeat as vlfeat

frames, descriptors = vlfeat.sift.dsift(image, step=5, fast=True)
descriptors.shape
descriptors[:5, :6]
#%%
from sklearn.cluster import MiniBatchKMeans
import numpy as np

X = np.array(
    [
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 0],
        [4, 4],
        [4, 5],
        [0, 1],
        [2, 2],
        [3, 2],
        [5, 5],
        [1, -1],
    ]
).astype(np.float32)
kmeans = MiniBatchKMeans(n_clusters=3, 
random_state=0, batch_size=2, verbose=1)
q = 3
# q = 4
# q = 2
# q = 6
for i in range(0, len(X), q):
    print(i)
    kmeans.partial_fit(X[i : i + q, :])
kmeans.cluster_centers_

# %%
assignments = vlfeat.kmeans.kmeans_quantize(X, kmeans.cluster_centers_)
assignments
# %%
np.histogram(assignments, bins=range(kmeans.n_clusters))[0]
# %%
kmeans.n_clusters


# %%
import numpy as np
import cv2 as cv
img = cv.imread(p)
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
# %%
kp[0].pt
des[0]
# %%
