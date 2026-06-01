from src.optimizers.optimizers import Adam

ALL_OBJECTS = {"adam": Adam}


def get(identifier):
    if isinstance(identifier, str):
        obj = ALL_OBJECTS.get(identifier)

    if callable(obj):
        return obj()

    raise ValueError(f"Could not interpret optimizer identifier: {identifier}")
