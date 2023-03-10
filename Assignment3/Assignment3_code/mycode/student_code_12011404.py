import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
from PIL import Image
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time


def get_tiny_images(image_paths):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    To build a tiny image feature, simply resize the original image to a very
    small square resolution, e.g. 16x16. You can either resize the images to
    square while ignoring their aspect ratio or you can crop the center
    square portion out of each image. Making the tiny images zero mean and
    unit length (normalizing them) will increase performance modestly.

    Useful functions:
    -   cv2.resize
    -   use load_image(path) to load a RGB images and load_image_gray(path) to
        load grayscale images

    Args:
    -   image_paths: list of N elements containing image paths

    Returns:
    -   feats: N x d numpy array of resized and then vectorized tiny images
              e.g. if the images are resized to 16x16, d would be 256
    """
    # dummy feats variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE
    # raise NotImplementedError('`get_tiny_images` function in ' +
    #       '`student_code.py` needs to be implemented')                                             #
    #############################################################################
    w, h = 16, 16
    N, d = len(image_paths), w * h
    feats = np.zeros((len(image_paths), d))

    def process(i, path):
        image = load_image_gray(path)
        feat = cv2.resize(image, (16, 16)).reshape(1, -1)
        feat = (feat - np.mean(feat)) / np.std(feat)
        feats[i, :] = feat

    pool = ThreadPool(16)
    pool.map(lambda i: process(i, image_paths[i]), range(N))
    pool.close()
    pool.join()
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


from sklearn.cluster import MiniBatchKMeans, KMeans

from threading import Lock
from queue import Queue
import time
from concurrent import futures
from tqdm import tqdm


def build_vocabulary(image_paths, vocab_size, threads=32, sift_param=None):
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
            -  frames is a N x 2 matrix of locations, which can be thrown away
            here (but possibly used for extra credit in get_bags_of_sifts if
            you're making a "spatial pyramid").
            -  descriptors is a N x 128 matrix of SIFT features
          Note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster to
          compute. Also, be sure not to use the default value of step size. It
          will be very slow and you'll see relatively little performance gain
          from extremely dense sampling. You are welcome to use your own SIFT
          feature Assignment4! It will probably be slower, though.
    -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
            http://www.vlfeat.org/matlab/vl_kmeans.html
              -  X is a N x d numpy array of sampled SIFT features, where N is
                 the number of features sampled. N should be pretty large!
              -  K is the number of clusters desired (vocab_size)
                 cluster_centers is a K x d matrix of cluster centers. This is
                 your vocabulary.

    Args:
    -   image_paths: list of image paths. ???????????????????????????
    -   vocab_size: size of vocabulary. ????????????????????????

    Returns:
    d??????????????????
    ????????????????????? ?????????????????????????????????????????????????????????
    -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
        cluster center / visual word
    """
    # Load images from the training set. To save computation time, you don't
    # necessarily need to sample from all images, although it would be better
    # to do so. You can randomly sample the descriptors from each image to save
    # memory and speed up the clustering. Or you can simply call vl_dsift with
    # a large step size here, but a smaller step size in get_bags_of_sifts.
    #
    # For each loaded image, get some SIFT features. You don't have to get as
    # many SIFT features as you will in get_bags_of_sift, because you're only
    # trying to get a representative sample here.
    #
    # Once you have tens of thousands of SIFT features from many training
    # images, cluster them with kmeans. The resulting centroids are now your
    # visual word vocabulary.

    # length of the SIFT descriptors that you are going to compute.
    dim = 128
    vocab = np.zeros((vocab_size, dim))
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # raise NotImplementedError('`build_vocabulary` function in ' +
    #       '`student_code.py` needs to be implemented')
    #############################################################################
    N = len(image_paths)
    # ????????????????????????????????? partial fit
    model = MiniBatchKMeans(n_clusters=vocab_size, init="k-means++", random_state=0)
    temp_lock = Lock()  # partial fit ???????????????
    # descriptor_channel = Queue(maxsize=threads * 2)
    descriptor_channel = Queue()

    def producer(i, path):
        image = load_image_gray(path)
        frames, descriptors = vlfeat.sift.dsift(
            image, step=5, fast=True, float_descriptors=True, norm=True
        )  # N x 128
        descriptor_channel.put(descriptors)

    producer_done = False

    def consumer():
        # print("consumer start")
        while not (producer_done and descriptor_channel.empty()):
            # print("consumer get")
            descriptors = descriptor_channel.get()
            # print("consumer get done")
            with temp_lock:
                model.partial_fit(descriptors)
            descriptor_channel.task_done()

    producer_tasks = []
    comsumer_tasks = []
    with futures.ThreadPoolExecutor(max_workers=len(image_paths) * 3) as executor:
        for i, path in enumerate(image_paths):
            producer_tasks.append(
                executor.submit(
                    producer,
                    i, path
                )
            )

        for i in range(len(image_paths) * 2):
            comsumer_tasks.append(
                executor.submit(
                    consumer
                )
            )

        print("producer_tasks", len(producer_tasks))
        for task in tqdm(futures.as_completed(producer_tasks), total=len(producer_tasks)):
            pass  # ????????????????????????
        producer_done = True
        print("producer_done: ", producer_done)

        for task in tqdm(futures.as_completed(comsumer_tasks), total=len(comsumer_tasks)):
            pass
    vocab = model.cluster_centers_

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return vocab


# from kmeans_pytorch import kmeans
# import torch
def build_vocabulary_no_parallel(image_paths, vocab_size):
    dim = 128
    descriptors_queue = []

    def process(i, path):
        nonlocal descriptors_queue
        image = load_image_gray(path)
        frames, descriptors = vlfeat.sift.dsift(
            image, step=5, fast=True, float_descriptors=True, norm=True
        )  # N x 128
        descriptors_queue += descriptors.tolist()

    paths = tqdm(enumerate(image_paths))
    for i, path in paths:
        paths.set_description(f"processing the {i}th image. ")
        process(i, path)

    # kmeans
    # cluster_ids_x, cluster_centers = kmeans(
    #     X=torch.Tensor(descriptors_queue), num_clusters=vocab_size, distance='euclidean', device=torch.device('cuda:0')
    # )
    # vocab = cluster_centers
    cluster_centers = vlfeat.kmeans.kmeans(np.array(descriptors_queue), vocab_size)
    vocab = cluster_centers
    return vocab


def build_vocabulary_parrallel(image_paths, vocab_size, threads=32, sift_param=None):
    dim = 128
    vocab = np.zeros((vocab_size, dim))
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # raise NotImplementedError('`build_vocabulary` function in ' +
    #       '`student_code.py` needs to be implemented')
    #############################################################################
    N = len(image_paths)
    # ????????????????????????????????? partial fit
    model = MiniBatchKMeans(n_clusters=vocab_size, init="k-means++", random_state=0)
    temp_lock = Lock()  # partial fit ???????????????

    def process(i, path):
        nonlocal model
        image = load_image_gray(path)
        frames, descriptors = vlfeat.sift.dsift(
            image, step=5, fast=True, float_descriptors=True, norm=True
        )  # N x 128
        with temp_lock:
            model = model.partial_fit(descriptors)

    tasks = []
    with futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for i, path in enumerate(image_paths):
            tasks.append(
                executor.submit(
                    process,
                    i, path
                )
            )
        for task in tqdm(futures.as_completed(tasks), total=len(tasks)):
            pass  # ????????????????????????
    # print(tasks[0].result())
    vocab = model.cluster_centers_
    return vocab


import multiprocessing


def get_bags_of_sifts(image_paths, vocab_filename, step=5, threads=32, vocab=None):
    """
    This feature representation is described in the handout, lecture
    materials, and Szeliski chapter 14.
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
            http://www.vlfeat.org/matlab/vl_dsift.html
          frames is a M x 2 matrix of locations, which can be thrown away here
            (but possibly used for extra credit in get_bags_of_sifts if you're
            making a "spatial pyramid").
          descriptors is a M x 128 matrix of SIFT features
            note: there are step, bin size, and smoothing parameters you can
            manipulate for dsift(). We recommend debugging with the 'fast'
            parameter. This approximate version of SIFT is about 20 times faster
            to compute. Also, be sure not to use the default value of step size.
            It will be very slow and you'll see relatively little performance
            gain from extremely dense sampling. You are welcome to use your own
            SIFT feature Assignment4! It will probably be slower, though.
    -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
            finds the cluster assigments for features in data
              -  data is a M x d matrix of image features
              -  vocab is the vocab_size x d matrix of cluster centers
              (vocabulary)
              -  assignments is a Mx1 array of assignments of feature vectors to
              nearest cluster centers, each element is an integer in
              [0, vocab_size)

    Args:
    -   image_paths: paths to N images
    -   vocab_filename: Path to the precomputed vocabulary.
            This function assumes that vocab_filename exists and contains an
            vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
            or visual word. This ndarray is saved to disk rather than passed in
            as a parameter to avoid recomputing the vocabulary every run.

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
            feature representation. In this case, d will equal the number of
            clusters or equivalently the number of entries in each image's
            histogram (vocab_size) below.
    """
    # load vocabulary
    if vocab is None:
        with open(vocab_filename, "rb") as f:
            vocab = pickle.load(f)

    dim = vocab.shape[0]  # ???????????????????????????????????????
    N = len(image_paths)
    # dummy features variable
    feats = np.zeros((N, dim))

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # raise NotImplementedError(
    # "`get_bags_of_sifts` function in " + "`student_code.py` needs to be implemented"
    # )
    #############################################################################
    def process_get_bags_of_sifts(i, path):
        image = load_image_gray(path)
        frames, descriptors = vlfeat.sift.dsift(
            image, step=5, fast=True, float_descriptors=True, norm=True
        )  # N x 128
        assignments = vlfeat.kmeans.kmeans_quantize(
            descriptors, vocab
        )  # M x 1. M???sift???????????????
        #  ?????????1?????????histogram???bins???????????????????????????????????????bin???????????????vocab_size
        hist, _ = np.histogram(
            assignments, bins=range(vocab.shape[0] + 1), density=True
        )  # ??????cluster?????????density??????hist???bins????????????1??? 
        feats[i, :] = hist

    tasks = []
    with futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for i, path in enumerate(image_paths):
            tasks.append(
                executor.submit(
                    process_get_bags_of_sifts,
                    i, path
                )
            )
        t_tasks = tqdm(futures.as_completed(tasks), total=len(tasks))
        for i, task in enumerate(t_tasks):
            t_tasks.set_description(f"processing the {i}th image. ")
            pass  # ????????????????????????

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


from scipy import stats


def nearest_neighbor_classify(
        train_image_feats, train_labels, test_image_feats, metric="euclidean", k=5
):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which will increase
    performance (although you need to pick a reasonable value for k).

    Useful functions:
    -   D = sklearn_pairwise.pairwise_distances(X, Y)
          computes the distance matrix D between all pairs of rows in X and Y.
            -  X is a N x d numpy array of d-dimensional features arranged along
            N rows
            -  Y is a M x d numpy array of d-dimensional features arranged along
            N rows
            -  D is a N x M numpy array where d(i, j) is the distance between row
            i of X and row j of Y

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating
            the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter Assignment4
    -   metric: (optional) metric to be used for nearest neighbor.
            Can be used to select different distance functions. The default
            metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
            well for histograms

    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE
    #     raise NotImplementedError('`nearest_neighbor_classify` function in ' +
    #       '`student_code.py` needs to be implemented')                                               #
    #############################################################################
    D = sklearn_pairwise.pairwise_distances(
        train_image_feats, test_image_feats, metric=metric, n_jobs=32
    )

    nearest = np.argsort(D, axis=1)[:, :k].astype(int)  # ????????????????????? ?????????k???????????????
    neighbour_labels = np.array(train_labels)[
        nearest
    ]  # ??????nearest????????????????????????train_labels?????????
    test_labels = stats.mode(
        neighbour_labels, axis=1, nan_policy="raise"
    ).mode.squeeze()
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels


def svm_classify(train_image_feats, train_labels, test_image_feats, threads=32):
    """
    This function will train a linear SVM for every category (i.e. one vs all)
    and then use the learned linear classifiers to predict the category of
    every test image. Every test feature will be evaluated with all 15 SVMs
    and the most confident SVM will "win". Confidence, or distance from the
    margin, is W*X + B where '*' is the inner product or dot product and W and
    B are the learned hyperplane parameters.

    Useful functions:
    -   sklearn LinearSVC
          http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    -   svm.fit(X, y)
    -   set(l)

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating the
            ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter Assignment4
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    # categories
    train_labels = np.array(train_labels)
    categories = list(set(train_labels))

    # construct 1 vs all SVMs for each category
    svms = {
        cat: LinearSVC(random_state=0,
                       tol=1e-3, loss="hinge", C=1.0)
        for cat in categories
    }

    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # raise NotImplementedError(
    # "`svm_classify` function in " + "`student_code.py` needs to be implemented"
    # )
    #############################################################################
    tasks, results = [], np.zeros((len(categories), test_image_feats.shape[0]))
    with futures.ThreadPoolExecutor(max_workers=threads) as executor:
        def process(i, cat):
            svms[cat].fit(train_image_feats, train_labels == cat)
            results[i, :] = svms[cat].decision_function(test_image_feats)

        for i, cat in enumerate(categories):
            tasks.append(
                executor.submit(
                    process,
                    i, cat
                )
            )
        for task in tqdm(futures.as_completed(tasks), total=len(tasks)):
            pass  # ????????????????????????
    test_labels = np.array(categories)[np.argmax(results, axis=0)]  # ??????????????????????????????????????????????????????
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels


## ??????sklearn??????
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABCMeta, abstractmethod
import joblib
from joblib import memory

memory = joblib.Memory('./tmp', verbose=0)

c_build_vocabulary_parrallel = memory.cache(build_vocabulary_parrallel)
c_get_bags_of_sifts = memory.cache(get_bags_of_sifts)
c_svm_classify = memory.cache(svm_classify)


class SiftClassifier(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self, vocab_size=100, step_size=10):
        self.parameters = {'vocab_size': vocab_size, 'step_size': step_size}

    def fit(self, X, y):
        self.parameters['vocab'] = c_build_vocabulary_parrallel(X, self.parameters['vocab_size'],
                                                                2 * self.parameters['step_size'])
        self.parameters['train_feats'] = c_get_bags_of_sifts(X, '', self.parameters['step_size'], 32,
                                                             self.parameters['vocab'])
        self.parameters['train_labels'] = y
        return self

    def get_params(self, deep=False):
        """???????????????????????????????????????????????????????????????sk??????????????????"""
        return self.parameters

    def set_params(self, **parameters):
        self.parameters = parameters

    def predict(self, X):
        test_feats = c_get_bags_of_sifts(X, '', self.parameters['step_size'], 32, self.parameters['vocab'])
        return c_svm_classify(self.parameters['train_feats'], self.parameters['train_labels'], test_feats)

    def predict_proba(self, X):
        raise NotImplementedError("predict_proba is not implemented")

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
