import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width, aperture_sobel=3):
    """
    检测在所有方向上强度变化都很大的图像区域，这些区域被称为角点(corner)。
    评价标准 E(u, v) = sum(W(x, y) * (I(x + u, y + v) - I(x, y))**2))
    u, v是特征点位置, xy是邻域偏移。
    Args:
    -   image: A numpy array of shape (m,n,c),
                    image may be grayscale of color (your choice)
                    值的亮度单位是0-1.0, 不是0-255
    -   feature_width: integer representing the local feature width in pixels. 局部特征的宽度
    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points 
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point (e.g. the Harris corner score) 用于描述每一个特征点的强度
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point (e.g. the radius of the circle that generated the interest) 
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point (单位为度) 用于描述每一个特征点的方向
    """
    ##############################################################################
    ##                            START OF get_interest_points                              #
    #############################################################################
    confidences, scales, orientations = None, None, None
    feature_width = feature_width // 2 *2 +1
    # 转换为灰度图像. 可能已经转过了，是二维的，就不转；可能是三维，但是只有一个通道，也不转。
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    # 标准化, 防止输入的有时候是0-255, 有时候是0-1.0。
    gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
    # 计算梯度
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=aperture_sobel)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=aperture_sobel)
    dx_2 = dx ** 2
    dx_dy = dx * dy
    dy_2 = dy ** 2
    # 卷积高斯窗口。 0表示自动根据feature_width计算sigma。公式是sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    g_dx_2 = cv2.GaussianBlur(dx_2, (feature_width, feature_width), 0)
    g_dx_dy = cv2.GaussianBlur(dx_dy, (feature_width, feature_width), 0)
    g_dy_2 = cv2.GaussianBlur(dy_2, (feature_width, feature_width), 0)
#     M = np.array([[g_dx_2, g_dx_dy], [g_dx_dy, g_dy_2]]) # 这个张量，不需要声明出来。
    # 现在 det(M) = ab trace(M) = a + b, ab为两个特征值。
    detM = g_dx_2 * g_dy_2 - g_dx_dy ** 2 # 对每一个像素点都有一个detM
    trM = g_dx_2 + g_dy_2
    confidences = detM - 0.04 * trM ** 2 # 这是Harris标准。SIFT的标准不一样。
#     scales = feature_width * np.ones_like(confidences)

    x,y = np.where(confidences > 0.01 * np.max(confidences)) # 0.01是阈值，可以调整。
    scales = np.ones_like(x) * feature_width
    ##############################################################################
    ##                            START adaptive non-maximal suppression                             #
    #############################################################################
    confidences = confidences[x, y]
    confidence_median = np.median(confidences)
    confidence_min = np.min(confidences)
    good = np.where(confidences > (confidence_median+confidence_min)/2)
    x = x[good]
    y = y[good]
    confidences = confidences[good]
    scales = scales[good]
    
    # sort the points by confidence
    sort = np.argsort(confidences)[::-1]
    confidences = confidences[sort]
    scales = scales[sort]
    x = x[sort]
    y = y[sort]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, confidences, scales, orientations

# def my_cornerHarris(gray_image, feature_width, k_size, k):
