from src.losses.losses import binary_crossentropy
from src.losses.losses import mean_squared_error

ALL_OBJECTS = {
    binary_crossentropy,
    mean_squared_error,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}


def get(identifier):
    if identifier is None:
        return None

    if isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        return obj
    else:
        raise ValueError(f"Could not interpret loss identifier: {identifier}")
