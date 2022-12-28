#%%
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
X = train_image_paths+test_image_paths
y = train_labels+test_labels

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
from tqdm import trange
def evaluate_model(X_train, y_train, vocab_size=100,  kfold = 5):
    # 用整个train数据集建立词表
    vocab = sc.c_build_vocabulary_parrallel(X_train, vocab_size)
    block_size = len(X_train)//kfold
    # 分出验证集
    # tr = trange(kfold)
    accs = []
    for i in range(kfold):
        X_real_train, X_val = X_train[:i*block_size]+X_train[(i+1)*block_size:], X_train[i*block_size:(i+1)*block_size]
        y_real_train, y_val = y_train[:i*block_size]+y_train[(i+1)*block_size:], y_train[i*block_size:(i+1)*block_size]
        # 用词表建立训练集的特征
        X_train_features = sc.c_get_bags_of_sifts(X_real_train, '', step=5, threads=32, vocab=vocab)
        # 用词表建立验证集的特征
        X_val_features = sc.c_get_bags_of_sifts(X_val, '', step=5, threads=32, vocab=vocab)
        # 训练模型 验证模型
        test_labels = sc.c_svm_classify(X_train_features, y_real_train, X_val_features)
        acc = np.mean(test_labels == y_val)
        # tr.set_description(f"kfold: {i}, acc: {acc}")
        print(f"kfold: {i}, acc: {acc}")
        accs.append(acc)
    return np.mean(accs)
# evaluate_model( X_test, y_test, vocab_size=100,  kfold = 5)

#%%
vocab_sizes = range(100,200, 100)
accs = []
for vocab_size in vocab_sizes:
    accs.append(evaluate_model( train_image_paths, train_labels, vocab_size=vocab_size,  kfold = 4))
plt.plot(vocab_sizes, accs)
# %%

a = sc.get_bags_of_sifts(X_test, 'vocab.pkl')

# %%
(a==0).all().all()

# %%


