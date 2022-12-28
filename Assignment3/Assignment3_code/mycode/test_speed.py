#%%
# Set up parameters, image paths and category list
# %matplotlib notebook
# %matplotlib widget 
%load_ext autoreload
%autoreload 2

import cv2
import numpy as np
import os.path as osp
import pickle
from random import shuffle
import matplotlib.pyplot as plt
from utils import *
import student_code_12011404 as sc


# This is the list of categories / directories to use. The categories are
# somewhat sorted by similarity so that the confusion matrix looks more
# structured (indoor and then urban and then rural).
categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial', 'Suburb',
              'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry', 'Coast',
              'Mountain', 'Forest'];
# This list of shortened category names is used later for visualization
abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst',
                   'Mnt', 'For'];

# Number of training examples per category to use. Max is 100. For
# simplicity, we assume this is the number of test cases per category, as
# well.
num_train_per_cat = 100

# This function returns lists containing the file path for each train
# and test image, as well as lists with the label of each train and
# test image. By default all four of these lists will have 1500 elements
# where each element is a string.
data_path = osp.join('..', 'data')
train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path,
                                                                                 categories,
                                                                                 num_train_per_cat)
                                                                                                                           
#%%
import joblib
from joblib import memory
memory = joblib.Memory('./tmp')
build = memory.cache(sc.build_vocabulary_parrallel)
测试写的对不对 = train_image_paths[:64]
build(测试写的对不对, 1).shape
build(测试写的对不对, 24).shape # 看来没问题                                                                                                                                                                                                                               num_train_per_cat);
#%%
# print('Using the BAG-OF-SIFT representation for images')
# 测试写的对不对 = train_image_paths[:1024]
# vocab_size = 200  # Larger values will work better (to a point) but be slower to compute
# vocab = sc.build_vocabulary_parrallel(测试写的对不对, vocab_size)
# # 10s 处理完sift特征
# # kmeans 很慢 1m28s

# # 普通parallel 可以 49s

#%%
# 测试写的对不对 = train_image_paths[:64]
# vocab_size = 200
# vocab = sc.build_vocabulary_no_parallel(测试写的对不对, vocab_size)
# # 42s sift 1024

# # 2s sift 64
# # 47s Kmeans vlf
#%%
# 200x200 的图片 ，sift一次9ms, 
# 1500张图片， 9ms * 1500 = 13.5s 。  很快
# 30s