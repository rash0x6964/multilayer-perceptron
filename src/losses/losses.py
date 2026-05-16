
import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()


def binary_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
