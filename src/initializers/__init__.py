from src.initializers.random_initializers import HeUniform
from src.initializers.random_initializers import Zeros


ALL_OBJECTS = {
    HeUniform,
    Zeros,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}


def get(identifier):
    if identifier is None:
        return None

    if isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier)
        if obj is not None:
            return obj()

    raise ValueError(
        f"Could not interpret initializer identifier: {identifier}"
    )
