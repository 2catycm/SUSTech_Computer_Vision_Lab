from concurrent import futures

import numpy as np
import cyvlfeat as vlfeat
from sklearn.model_selection import train_test_split, cross_val_score
from tqdm import tqdm, trange

from .augmentation import Augmentor
from .utils import *
import os.path as osp
from glob import glob
from random import shuffle
from sklearn.svm import LinearSVC

import cv2
import math

from threading import Lock

import joblib

memory = joblib.Memory('./joblib_tmp', verbose=1)

vl_hog = vlfeat.hog


def get_positive_features(train_path_pos, feature_params, threads=32):
    """
    This function should return all positive training examples (faces) from
    36x36 images in 'train_path_pos'. Each face should be converted into a
    HoG template according to 'feature_params'.
    从36x36的图像中提取所有的正样本（是人脸的样本），
    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
            注意这里的计算，是将36x36的图像，分成6x6的小块，每个小块有31个特征，所以总共有36/6*36/6*31=1764个特征
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)  # 图像的大小，改不了。
    cell_size = feature_params.get('hog_cell_size', 6)  # 一个grid cell的大小。是36的因数。
    num_orientations = feature_params.get('num_orientations ', 9)
    bilinear_interpolation = feature_params.get('bilinear_interpolation ', False)
    augmentation_num = feature_params.get('augmentation_num', 4)
    return get_positive_features_impl(train_path_pos, win_size, cell_size, num_orientations,
                          bilinear_interpolation, augmentation_num, threads=32)


@memory.cache
def get_positive_features_impl(train_path_pos, win_size, cell_size, num_orientations,
                          bilinear_interpolation, augmentation_num,
                          threads=32):
    positive_files = glob(osp.join(train_path_pos, '*.jpg'))
    ###########################################################################
    #                           YOUR CODE HERE                          #
    ###########################################################################

    n_cell = np.ceil(win_size / cell_size).astype('int')

    N = len(positive_files)

    augmentor = Augmentor(augmentation_num)
    feats = np.zeros((N * augmentation_num, n_cell * n_cell * 31), np.float32)

    def process(i, file):
        # 使用数据增强。将原始数据进行旋转，平移，缩放，镜像等操作，得到更多的正样本
        for j, image in enumerate(augmentor.gen_augmented_image(cv2.imread(file))):
            # 使用HOG特征提取器，提取特征。
            hog_feature = vl_hog.hog(cv_image2vl_image(image),
                                     cell_size=cell_size,
                                     n_orientations=num_orientations,
                                     bilinear_interpolation=bilinear_interpolation)
            index = i * augmentation_num + j
            feats[index] = hog_feature.reshape(-1)

    tasks = []
    with futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # 遍历所有的正样本。注意不是遍历train_path_pos，而是glob的结果——positive_files。
        for i, path in enumerate(positive_files):
            tasks.append(
                executor.submit(
                    process,
                    i, path
                )
            )
        for task in tqdm(futures.as_completed(tasks), total=len(tasks)):
            pass  # 等待所有任务完成
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return feats


def cv_image2vl_image(image):
    """
    This function converts an image from cv2 format to vlfeat format.
    Args:
    -   image: H x W x 3 image in cv2 format
    Returns:
    -   image: H x W x 3 image in vlfeat format
    """
    return image[:, :, ::-1].astype(np.float32)


def get_random_negative_features(non_face_scn_path, feature_params, num_samples, threads=32):
    """
        This function should return negative training examples (non-faces) from any
        images in 'non_face_scn_path'. Images should be loaded in grayscale because
        the positive training data is only available in grayscale (use
        load_image_gray()).

        Useful functions:
        -   vlfeat.hog.hog(im, cell_size): computes HoG features

        Args:
        -   non_face_scn_path: string. This directory contains many images which
                have no faces in them.
        -   feature_params: dictionary of HoG feature computation parameters. See
                the documentation for get_positive_features() for more information.
        -   num_samples: number of negatives to be mined. It is not important for
                the function to find exactly 'num_samples' non-face features. For
                example, you might try to sample some number from each image, but
                some images might be too small to find enough.

        Returns:
        -   N x D matrix where N is the number of non-faces and D is the feature
                dimensionality, which would be (feature_params['template_size'] /
                feature_params['hog_cell_size'])^2 * 31 if you're using the default
                hog parameters.
        """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    num_orientations = feature_params.get('num_orientations ', 9)  # 注意名字和hog里面不一样。
    bilinear_interpolation = feature_params.get('bilinear_interpolation ', False)

    return get_random_negative_features_impl(non_face_scn_path, win_size, cell_size, num_orientations,
                                        bilinear_interpolation, num_samples, threads=threads)


# 使用memory cache注意debug的时候要注释掉，不然会用上次的错误结果。
@memory.cache
def get_random_negative_features_impl(non_face_scn_path, win_size, cell_size, num_orientations,
                                 bilinear_interpolation, num_samples, threads=32):
    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           YOUR CODE HERE                          #
    ###########################################################################

    n_cell = np.ceil(win_size / cell_size).astype('int')
    N = len(negative_files)

    # sample_per_files = num_samples // N
    sample_per_files = num_samples  # 每个都10000个才好

    feats = np.zeros((sample_per_files * N, n_cell * n_cell * 31), np.float32)

    def process(i, file):
        image = load_image_gray(file)
        # 如果不够大，就不能采样，先变大。
        h, w = image.shape
        if h < win_size or w < win_size:  # 实际上没有这样的样本。
            image = cv2.resize(image, (max(win_size, h), max(win_size, w)))
            # 随机采样
        for j in range(sample_per_files):
            x = np.random.randint(0, w - win_size)
            y = np.random.randint(0, h - win_size)
            patch = image[y:y + win_size, x:x + win_size]
            hog_feature = vlfeat.hog.hog(patch,
                                         cell_size=cell_size,
                                         n_orientations=num_orientations,
                                         bilinear_interpolation=bilinear_interpolation
                                         )
            index = i * sample_per_files + j  # 注意不是num_samples
            feats[index] = hog_feature.reshape(-1)

    tasks = []
    with futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # 遍历所有的正样本。注意不是遍历train_path_pos，而是glob的结果——positive_files。
        for i, path in enumerate(negative_files):
            tasks.append(
                executor.submit(
                    process,
                    i, path
                )
            )
        for task in tqdm(futures.as_completed(tasks), total=len(tasks)):
            pass  # 等待所有任务完成
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats


def cross_validate_classifier(features_pos, features_neg, C):
    ###########################################################################
    #                          My CODE HERE                          #
    ###########################################################################
    svm = LinearSVC(C=C)
    X = np.vstack((features_pos, features_neg))
    y = np.hstack((np.ones(features_pos.shape[0]), np.zeros(features_neg.shape[0])))
    # 如果不shuffle，那么正负样本会在一起，导致训练集和测试集都是正样本。
    # 但是sklearn默认自带shuffle，所以不用担心。
    # Xy = np.hstack((X, y.reshape(-1, 1)))
    # np.random.shuffle(Xy)
    # X = Xy[:, :-1]
    # y = Xy[:, -1]
    # 5折交叉验证
    cv = 10
    n_jobs = -1
    # scores = cross_val_score(svm, X, y, scoring='recall', cv=cv, n_jobs=n_jobs)
    # scores = cross_val_score(svm, X, y, scoring='roc_auc', cv=cv, n_jobs=n_jobs)
    scores = cross_val_score(svm, X, y, scoring='average_precision', cv=cv, n_jobs=n_jobs)
    # scores = cross_val_score(svm, X, y, scoring='balanced_accuracy_score', cv=cv, n_jobs=n_jobs)
    # scores = cross_val_score(svm, X, y, scoring='f1', cv=10, n_jobs=n_jobs)
    return scores.mean()


@memory.cache
def train_classifier(features_pos, features_neg, C):
    """
    This function trains a linear SVM classifier on the positive and negative
    features obtained from the previous steps. We fit a model to the features
    and return the svm object.

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """
    ###########################################################################
    #                          YOUR CODE HERE                          #
    ###########################################################################

    svm = LinearSVC(C=C)
    X = np.vstack((features_pos, features_neg))
    y = np.hstack((np.ones(features_pos.shape[0]), np.zeros(features_neg.shape[0])))
    svm.fit(X, y)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm


def mine_hard_negs(non_face_scn_path, svm, feature_params, num_samples, threads=32):
    """
    This function is pretty similar to get_random_negative_features(). The only
    difference is that instead of returning all the extracted features, you only
    return the features with false-positive prediction.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    -   svm.predict(feat): predict features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """
    # My Assignment4 starts
    feats = get_random_negative_features(non_face_scn_path, feature_params, num_samples, threads=threads)
    y_pred = svm.predict(feats)
    return feats[y_pred == 1]  # 预测为1，实际上都是0，所以是false positive， 为hard negative。


def run_detector(test_scn_path, svm, feature_params, verbose=False, threads=32, im_filenames=None):
    """
        This function returns detections on all of the images in a given path. You
        will want to use non-maximum suppression on your detections or your
        performance will be poor (the evaluation counts a duplicate detection as
        wrong). The non-maximum suppression is done on a per-image basis. The
        starter Assignment4 includes a call to a provided non-max suppression function.

        The placeholder version of this Assignment4 will return random bounding boxes in
        each test image. It will even do non-maximum suppression on the random
        bounding boxes to give you an example of how to call the function.

        Your actual Assignment4 should convert each test image to HoG feature space with
        a _single_ call to vlfeat.hog.hog() for each scale. Then step over the HoG
        cells, taking groups of cells that are the same size as your learned
        template, and classifying them. If the classification is above some
        confidence, keep the detection and then pass all the detections for an
        image to non-maximum suppression. For your initial debugging, you can
        operate only at a single scale and you can skip calling non-maximum
        suppression. Err on the side of having a low confidence threshold (even
        less than zero) to achieve high enough recall.

        Args:
        -   test_scn_path: (string) This directory contains images which may or
                may not have faces in them. This function should work for the
                MIT+CMU test set but also for any other images (e.g. class photos).
        -   svm: A trained sklearn.svm.LinearSVC object
        -   feature_params: dictionary of HoG feature computation parameters.
            You can include various parameters in it. Two defaults are:
                -   template_size: (default 36) The number of pixels spanned by
                each train/test template.
                -   hog_cell_size: (default 6) The number of pixels in each HoG
                cell. template size should be evenly divisible by hog_cell_size.
                Smaller HoG cell sizes tend to work better, but they make things
                slowerbecause the feature dimensionality increases and more
                importantly the step size of the classifier decreases at test time.
        -   verbose: prints out debug information if True

        Returns:
        -   bboxes: N x 4 numpy array. N is the number of detections.
                bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
        -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
                confidence of detection i.
        -   image_ids: List with N elements. image_ids[i] is the image file name
                for detection i. (not the full path, just 'albert.jpg')
        """
    # number of top detections to feed to NMS
    # topk = 15
    topk = feature_params.get('topk', 150)
    # topk = 1500
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    # cell_size = feature_params.get('hog_cell_size', 4)
    num_orientations = feature_params.get('num_orientations ', 9)  # 注意名字和hog里面不一样。
    bilinear_interpolation = feature_params.get('bilinear_interpolation ', False)

    # scale_factor = feature_params.get('scale_factor', 0.65)
    scale_factor = feature_params.get('scale_factor', 0.9)
    basic_scales = max(feature_params.get('basic_scales', 1), 1)  # 至少是1
    # basic_scales = max(feature_params.get('basic_scales', 2), 0)  # 至少是1

    sigma0 = feature_params.get('sigma0', 1.52)
    # sigma0 = feature_params.get('sigma0', 0.2)
    # svm_threshold = feature_params.get('svm_threshold', 0)
    svm_threshold = feature_params.get('svm_threshold', -4)
    # svm_threshold = feature_params.get('svm_threshold', -2)
    # svm_threshold = feature_params.get('svm_threshold', -10)

    basic_octaves = feature_params.get('basic_octaves', 1)
    return run_detector_impl(test_scn_path, svm, topk, win_size, cell_size, num_orientations, bilinear_interpolation, scale_factor,
                 basic_scales, sigma0, svm_threshold, basic_octaves,
                             verbose=verbose, threads=threads, im_filenames=im_filenames)


@memory.cache
def run_detector_impl(test_scn_path, svm, topk, win_size, cell_size, num_orientations, bilinear_interpolation, scale_factor,
                 basic_scales, sigma0, svm_threshold, basic_octaves,
                 verbose=False, threads=32, im_filenames=None):
    if im_filenames is None:
        im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []

    # params for HOG computation
    template_size = int(win_size / cell_size)
    #######################################################################
    #                         YOUR CODE HERE                         #
    #######################################################################

    scales = basic_scales + 3
    # scales = basic_scales + 0
    k = (1 / scale_factor) ** (1 / basic_scales)

    # lock = Lock()
    # 准备遍历所有图片
    bar = tqdm(enumerate(im_filenames))

    def process_single_image(idx, im_filename):
        nonlocal bboxes, confidences, image_ids, bar
        # 本函数对单张图片进行检测
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape

        cur_x_min = np.empty(0, np.int)
        cur_y_min = np.empty(0, np.int)
        cur_bboxes = np.empty((0, 4), np.int)
        cur_confidences = np.empty(0, np.float)

        # create scale space HOG pyramid and return scores for prediction

        #######################################################################
        #                         YOUR CODE HERE                         #
        #######################################################################
        # 至少两个。除win_size表示octave最多放缩到和win_size一样大。
        # debug经验：im_shape.min()是错的，因为是tuple类型。应该是min(im_shape)。
        # octaves = max(feature_params.get('octaves', 0), 1) # 至少一次。

        octaves = max(
            int(math.log2(min(im_shape) / win_size) / math.log2(1 / scale_factor)) + basic_octaves,
            1)  # 至少一次。
        octave_base_image = im  # 基准图像
        if min(im_shape) < win_size:
            octave_base_image = cv2.resize(im, (max(im_shape[0], win_size), max(im_shape[1], win_size)))

        def process_single_scale(octave_base_image, octave, scale):
            nonlocal cur_x_min, cur_y_min, cur_bboxes, cur_confidences, bar
            sigma = sigma0 * (k ** scale)  # 不需要+octave*basic_scales, 因为后面的cv resize会自动blur。
            blured_im = cv2.GaussianBlur(octave_base_image, (0, 0), sigma)
            # blured_im = octave_base_image
            relative = ((1 / scale_factor) ** octave)

            hog_whole_image = vl_hog.hog(blured_im,
                                         cell_size=cell_size,
                                         n_orientations=num_orientations,
                                         bilinear_interpolation=bilinear_interpolation)
            # template_size 就是在hog空间上窗口的大小
            rows = max(hog_whole_image.shape[0] - template_size, 1)
            cols = max(hog_whole_image.shape[1] - template_size, 1)
            feats = np.empty((rows * cols, template_size * template_size * 31), float)
            for i in range(rows):
                for j in range(cols):
                    win_hog_feat = hog_whole_image[i:i + template_size, j:j + template_size, :]
                    feats[i * cols + j] = win_hog_feat.reshape(1, -1)
            confs = svm.decision_function(feats)
            is_face = confs > svm_threshold
            grid_indexes = np.nonzero(confs > svm_threshold)[0]
            cur_x_hog, cur_y_hog = grid_indexes // cols, grid_indexes % cols  # 注意都是cols。
            cur_y_min, cur_x_min = cur_x_hog.reshape(-1, 1) * cell_size, cur_y_hog.reshape(-1,
                                                                                           1) * cell_size  # 注意题目验证代码给的xy弄反了。所以我们只好正过来。
            cur_bboxes = (np.hstack(
                (cur_x_min, cur_y_min, cur_x_min + win_size, cur_y_min + win_size)) * relative).astype(int)
            cur_x_min, cur_y_min = cur_bboxes[:, 0], cur_bboxes[:, 1]
            cur_confidences = confs[is_face]

        octave_bases = []
        for octave in range(octaves):
            octave_bases.append(octave_base_image.copy())
            octave_base_image = cv2.resize(octave_base_image, (0, 0), fx=scale_factor,
                                           fy=scale_factor)  # resize对float没有问题。
        for octave in range(octaves):
            for scale in range(scales):
                process_single_scale(octave_bases[octave], octave, scale)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

        originally_found = len(cur_confidences)

        idsort = np.argsort(-cur_confidences)[:topk]
        cur_bboxes = cur_bboxes[idsort]
        cur_confidences = cur_confidences[idsort]

        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
                                                 im_shape, verbose=verbose)
        bar.set_description(
            'Detecting faces in {:s}. With octaves{:d}, Found {:d} box originally, after NMS, {:d} detections passed'.format(
                im_filename, octaves, originally_found, sum(is_valid_bbox)))
        # 如果没有box，就不需要做任何事情。做了会报错。
        if sum(is_valid_bbox) == 0:
            return

        # 过滤
        cur_bboxes = cur_bboxes[is_valid_bbox]
        cur_confidences = cur_confidences[is_valid_bbox]

        bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))

    for idx, im_filename in bar:
        process_single_image(idx, im_filename)

    return bboxes, confidences, image_ids
