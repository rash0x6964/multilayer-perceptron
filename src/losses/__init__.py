# from src.losses.losses import binary_crossentropy
# from src.losses.losses import mean_squared_error
from src.losses.losses import CategoricalCrossentropy

ALL_OBJECTS = {
    # binary_crossentropy,
    # mean_squared_error,
    "categorical_crossentropy": CategoricalCrossentropy
}

# ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}


def get(identifier):
    if identifier not in ALL_OBJECTS:
        raise ValueError(f"Could not interpret loss identifier: {identifier}")

    return ALL_OBJECTS[identifier]()
