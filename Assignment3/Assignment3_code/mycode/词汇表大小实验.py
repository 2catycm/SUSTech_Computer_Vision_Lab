# %%
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
categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial', 'Suburb',
              'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry', 'Coast',
              'Mountain', 'Forest'];
abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst',
                   'Mnt', 'For'];
num_train_per_cat = 100
data_path = osp.join('..', 'data')
train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path,
                                                                                 categories,
                                                                                 num_train_per_cat)

# %%
def evaluate_model(X_train, y_train, vocab_size=100,  kfold = 5):
    # 用整个train数据集建立词表
    vocab = sc.c_build_vocabulary_parrallel(X_train, vocab_size)
    block_size = len(X_train)//kfold
    # 分出验证集
    # tr = trange(kfold)
    accuracies = []
    feats = sc.c_get_bags_of_sifts(X_train, '', step=5, threads=32, vocab=vocab)
    for i in range(kfold):
        # 用词表建立训练集的特征
        X_train_features = np.concatenate((feats[:i*block_size], feats[(i+1)*block_size:]), axis=0)
        y_train_labels = np.concatenate((y_train[:i*block_size],y_train[(i+1)*block_size:]), axis=0)
        X_val_features = feats[i*block_size:(i+1)*block_size]
        y_val_labels = y_train[i*block_size:(i+1)*block_size]
        # 训练模型 验证模型
        y_val_labels_pred = sc.c_svm_classify(X_train_features, y_train_labels, X_val_features)
        acc = np.mean(y_val_labels == y_val_labels_pred)
        # tr.set_description(f"kfold: {i}, acc: {acc}")
        print(f"kfold: {i}, acc: {acc}")
        accuracies.append(acc)
    return np.mean(accuracies)

# %%
import random
vocab_sizes = range(100,1001, 200)
# vocab_sizes = range(10, 100, 20)
accs = []
N = len(train_image_paths)
for vocab_size in vocab_sizes:
    accs.append(evaluate_model( random.choices(train_image_paths, k=N//5), random.choices(train_labels, k=N//5), vocab_size=vocab_size,  kfold = 5))
plt.plot(vocab_sizes, accs)

# %%
accs

# %%
accuracies

# %%
evaluate_model( train_image_paths[:N//10], train_labels[:N//10], vocab_size=100,  kfold = 5)
# evaluate_model( train_image_paths[:N//5], train_labels[:N//5], vocab_size=300,  kfold = 5)

# %%
