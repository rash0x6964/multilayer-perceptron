
import numpy as np


# def mean_squared_error(y_true, y_pred):
#     return np.square(np.subtract(y_true, y_pred)).mean()


# def binary_crossentropy(y_true, y_pred):
#     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class CategoricalCrossentropy:
    def __init__(self) -> None:
        pass

    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def gradient(self, y_true, y_pred):
        return y_pred - y_true
