from src.layers.layer import Layer
from src import activations
from src import initializers
import numpy as np


class Dense(Layer):
    def __init__(self, units, activation=None, kernel_initializer="glorot_uniform", input_shape=None):
        super().__init__()
        self.units = units
        self.input_shape = input_shape 
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape=None):
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
        return self.activation(np.matmul(inputs, self.kernel) + self.bias)
