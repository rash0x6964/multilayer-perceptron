from src.optimizers.optimizers import sgd

ALL_OBJECTS = {
    sgd,
}

ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}


def get(identifier):
    if identifier is None:
        return None

    if isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        return obj

    raise ValueError(f"Could not interpret optimizer identifier: {identifier}")
