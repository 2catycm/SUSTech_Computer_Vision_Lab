from functools import lru_cache

import cv2
import numpy as np


def get_features(image, x, y, feature_width, scales=None, orientations=None, window_width=4, num_bins=8):
    """
    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
        注意理解几个参数的不同：
        feature_width ：特征宽度，表示子窗口的大小。比如论文图片中2*2个子区域, 每个子区域的宽度就是feature_width。这个一般是4的倍数。
        scales: 特征点被检测出来是特征时，高斯金字塔的尺度。在计算梯度的时候，按照论文要求，需要在高斯模糊金字塔的基础上去计算进梯度。
        window_width： 窗口宽度，表示子窗口的个数。比如论文图片中2*2个子区域，那么窗口宽度就是2。
        num_bins = 8： 梯度方向的个数。因为我们的feature_width指定的位置内一般是16倍的像素数，所以8个方向不会太多也不会太少。
        两次高斯平滑：第一个平滑是高斯模糊金字塔；第二个高斯平滑是加权，让特征描述更加稳定。

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
#     assert image.ndim == 2, 'Image must be grayscale'
    feature_width = min(4, feature_width/4)
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    scales = np.ones_like(x) if scales is None else scales
    orientations = np.zeros_like(x) if orientations is None else orientations
    feat_dim = (window_width ** 2) * num_bins
    fv = np.zeros((x.shape[0], feat_dim))
    #############################################################################
    #                           START OF get_interest_points                    #
    #############################################################################
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    for i, (a, b) in enumerate(zip(x, y)):  # a,b允许是浮点数，表示特征点的位置。
        scale = scales[i]  # 特征点的尺度。
        gdx = lets_blur(dx, scale)
        gdy = lets_blur(dy, scale)  # 根据卷积的交换律，这和先做高斯金字塔再求导是一样的。
        orientation = orientations[i]
        # 计算每个点的梯度方向和梯度幅值, 其实就是把笛卡尔坐标转换成极坐标。
        mag, ang = lets_cartToPolar(gdx, gdy, orientation)
        # 接下来，用 window_width 去找 feature_width 宽度的子区域，然后计算梯度方向的直方图。
        real_width = feature_width * window_width
        leftup_x = a - real_width / 2  # 是浮点数，我们再实际取值的时候，才取int。而且可以插值。
        leftup_y = b - real_width / 2
        for q, (subwin_i, subwin_j) in enumerate(np.ndindex(window_width, window_width)):
            subwin_x = leftup_x + subwin_i * feature_width
            subwin_y = leftup_y + subwin_j * feature_width
            x_s, x_e, y_s, y_e = tuple(map(
                int, (subwin_x, subwin_x + feature_width, subwin_y, subwin_y + feature_width)))
            subwindow_mag = mag[x_s:x_e, y_s:y_e]
            # 计算梯度方向的直方图
            # 单位为度
            hist, _ = np.histogram(ang[x_s:x_e, y_s:y_e], bins=num_bins, range=(0, 360),
                                   weights=subwindow_mag)
            fv[i, q*num_bins:(q+1)*num_bins] = hist
        if i%200 == 0:
            print(f"getting the {i}th/{x.shape[0]} feature vector\r")
        #############################################################################
        #                             END OF get_interest_points                             #
        #############################################################################
    # 归一化
    fv = fv / np.linalg.norm(fv, axis=1, keepdims=True)  # 让向量L2范数为1
    return fv


def hash_nparray(arr):
    return hash(str(arr))


results_cartToPolar = {}


def lets_cartToPolar(dx, dy, real_width=4*4, orientation=0):
    signature = (hash_nparray(dx), hash_nparray(dy), real_width, orientation)
    if signature in results_cartToPolar:
        return results_cartToPolar[signature]

    mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    mag = lets_blur(mag, 1.5*real_width, True)  # 是为了让特征更加稳定。
    ang = orientation-ang  # 旋转到主方向, 以提供旋转不变性。
    ang = (ang+360) % 360  # 旋转到0-360度

    results_cartToPolar[signature] = (mag, ang)
    return mag, ang


# @lru_cache(maxsize=4)  # 因为可能scales里面传进来的几乎都是一样的。（上一步用的是harris算法，没有改scale）
results_blur = {}


def lets_blur(mat, scale, no_cache=False):
    signature = (hash_nparray(mat), scale)
    if signature in results_blur and not no_cache:
        return results_blur[signature]
    ksize = int(2*np.ceil(3*scale) + 1) // 2 * 2 + 1  # 3sigma原则
    # 自动根据scale计算高斯核的大小。太远的就不要了。
    res = cv2.GaussianBlur(mat, (ksize, ksize), scale)
    if not no_cache:
        results_blur[signature] = res
    return res
