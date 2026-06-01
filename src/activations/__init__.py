from src.activations.activations import softmax
from src.activations.activations import softmax_grad
from src.activations.activations import sigmoid
from src.activations.activations import sigmoid_grad


ALL_OBJECTS_DICT = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "softmax": (softmax, softmax_grad),
}


def get(identifier):
    if isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier)

    if obj:
        return obj

    raise ValueError(
        f"Could not interpret initializer identifier: {identifier}"
    )
