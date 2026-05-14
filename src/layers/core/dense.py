from src.layers.layer import Layer
from src import activations
import numpy as np


class Dense(Layer):
    def __init__(self, units, activation=None, kernel_initializer="glorot_uniform"):
        super().__init__(act)
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernal",
        )
        self.bias = self.add_weight(
            shape=(self.units),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        return np.matmul(inputs, self.kernel) + self.bias
