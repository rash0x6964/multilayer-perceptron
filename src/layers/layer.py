import numpy as np


class Layer:
    def __init__(self, name=None):
        ...

    def __call__(self):
        return self.call()

    def call(self, ):
        raise NotImplementedError

    def add_weight(self, shape=None, initializer=None, trainable=True, name=None):
        ...
