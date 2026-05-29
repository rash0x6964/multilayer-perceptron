import math
import numpy as np


# def initialize_he(fan_in, fan_out):
#     # He Normal``
#     std = np.sqrt(2 / fan_in)
#     weights = np.random.normal(0, std, (fan_in, fan_out))
#     biases = np.zeros((1, fan_out))
#     return weights, biases


class HeUniform:
    def __init__(self, scale=2.0, mode="fan_in", distribution="uniform", seed=None):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape):
        scale = self.scale
        fan_in, fan_out = compute_fans(shape)

        scale /= max(1.0, fan_in)

        limit = math.sqrt(3.0 * scale)
        return np.random.uniform(low=-limit, high=limit, size=shape)


class Zeros:
    def __init__(self):
        pass

    def __call__(self, shape):
        return np.zeros(shape)


def compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
        shape: Integer shape tuple.

    Returns:
        A tuple of integer scalars: `(fan_in, fan_out)`.
    """
    shape = tuple(shape)
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)
