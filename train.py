import src
from src import layers
import numpy as np

np.random.default_rng(42)

input = np.random.random((32, 10))
input_shape = input.shape
output_shape = (2,)

model = src.Sequential([
    layers.Dense(input_shape, activation='sigmoid'),
    layers.Dense(24, activation='sigmoid', kernel_initializer='heUniform'),
    layers.Dense(24, activation='sigmoid', kernel_initializer='heUniform'),
    layers.Dense(24, activation='sigmoid', kernel_initializer='heUniform'),
    layers.Dense(output_shape, activation='softmax',
                 kernel_initializer='heUniform'),
])
