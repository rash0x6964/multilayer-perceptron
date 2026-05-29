import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_grad(a):
    return np.ones_like(a)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(a):
    return a * (1.0 - a)


# def relu(x):
#     np.maximum(0, x)
