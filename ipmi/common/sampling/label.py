from random import shuffle

import numpy as np


def uniform_train_test_splitting(data, labels, n_samples=100):
    index = list(range(len(data)))
    shuffle(index)
    split_data = []
    split_label = []
    rest_data = []
    rest_label = []
    counter = np.zeros(max(labels) + 1)
    for i in index:
        label = labels[i]
        if counter[label] < n_samples:
            split_label.append(label)
            split_data.append(data[i])
            counter[label] = counter[label] + 1
        else:
            rest_label.append(label)
            rest_data.append(data[i])
    return split_data, split_label, rest_data, rest_label


def uniform_subsampling(data, labels, n_samples=100):
    index = list(range(len(data)))
    shuffle(index)
    new_data = []
    new_label = []
    counter = np.zeros(max(labels) + 1)
    for i in index:
        label = labels[i]
        if counter[label] < n_samples:
            new_label.append(label)
            new_data.append(data[i])
            counter[label] = counter[label] + 1
    return new_data, new_label


def uniform_multilabel_train_test_splitting(data, labels, n_samples=100):
    index = list(range(len(data)))
    shuffle(index)
    split_data = []
    split_label = []
    rest_data = []
    rest_label = []
    counter = np.zeros(len(labels[0]))
    for i in index:
        label = labels[i]
        if ((counter < n_samples) & label.astype(bool)).any():
            split_label.append(label)
            split_data.append(data[i])
            counter += label
        else:
            rest_label.append(label)
            rest_data.append(data[i])
    return split_data, split_label, rest_data, rest_label
