from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

import time


image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


pkl_file = open('notMNIST.pickle', 'rb')
all_datasets = pickle.load(pkl_file)
train_dataset = all_datasets["train_dataset"]
valid_dataset = all_datasets["valid_dataset"]
test_dataset = all_datasets["test_dataset"]

train_labels = all_datasets["train_labels"]
valid_labels = all_datasets["valid_labels"]
test_labels = all_datasets["test_labels"]


sizes = [50, 100, 1000, 5000]
y_valid = np.reshape(valid_dataset, newshape=(len(valid_dataset), image_size*image_size))


print("=== LogClassifier results on non-sanitized sets ===")
for size in sizes:
    X = np.reshape(train_dataset[0:size], newshape=(size, image_size*image_size))
    log_reg = LogisticRegression(random_state=42)
    t0 = time.time()
    log_reg.fit(X, train_labels[0:size])
    t1 = time.time()
    print("train time for size '{}': {:.4f}".format(size, t1-t0))
    score = log_reg.score(y_valid, valid_labels)
    print("score for size '{}': {:.4f}".format(size, score))
