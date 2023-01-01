import numpy as np
import scipy
from scipy import spatial


def match_features(features1, features2, x1, y1, x2, y2):
    """
    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    matches = []
    confidences = []
    #############################################################################
    #                           START OF match_features                         #
    #############################################################################
    tree = spatial.KDTree(features1)
    for i, feat in enumerate(features2):
        dist, index = tree.query(feat, 1)
        matches.append([index, i])
        confidences.append(dist)
    #############################################################################
    #                             END OF match_features                              #
    #############################################################################
    confidences = np.array(confidences)
    matches = np.array(matches)
    confidence_median = np.median(confidences)
    confidence_min = np.min(confidences)
    good = np.where(confidences > (confidence_median+confidence_min)/2)
    matches = matches[good]
    confidences = confidences[good]
    return matches, confidences


