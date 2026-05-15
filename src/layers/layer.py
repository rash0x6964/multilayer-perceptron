import numpy as np
from src import initializers


class Layer:
    def __init__(self, name=None):
        ...

    def __call__(self):
        return self.call()

    def call(self, ):
        raise NotImplementedError

    def add_weight(self, shape=None, initializer=None, trainable=True, name=None):
        if shape is None:
            shape = ()
        if initializer is None:
            initializer = "glorot_uniform"
        initializer = initializers.get(initializer)

        if initializer is None:
            return None

        return initializer()(shape)
