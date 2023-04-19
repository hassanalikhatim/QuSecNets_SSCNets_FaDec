import matplotlib.pyplot as plt
import numpy as np
import os


def confirm_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def permute(X, y):
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y