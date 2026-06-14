from src.losses.losses import CategoricalCrossentropy
from src.losses.losses import BinaryCrossentropy

ALL_OBJECTS = {
    "categorical_crossentropy": CategoricalCrossentropy,
    "binary_crossentropy": BinaryCrossentropy,
}

# ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}


def get(identifier):
    if identifier not in ALL_OBJECTS:
        raise ValueError(f"Could not interpret loss identifier: {identifier}")

    return ALL_OBJECTS[identifier]()
