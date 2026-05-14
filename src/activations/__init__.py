from src.activations.activations import relu
from src.activations.activations import softmax
from src.activations.activations import sigmoid
from src.activations.activations import linear

ALL_OBJECTS = {
    relu,
    softmax,
    sigmoid,
    linear,
}

ALL_OBJECTS_DICT = {fn.__name__: fn for fn in ALL_OBJECTS}


def get(identifier):
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier)
    if callable(obj):
        return obj

    raise ValueError(
        f"Could not interpret activation function identifier: {identifier}"
    )
