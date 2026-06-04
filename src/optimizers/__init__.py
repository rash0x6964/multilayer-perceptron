from src.optimizers.optimizers import Optimizer
from src.optimizers.optimizers import Adam
from src.optimizers.optimizers import SGD

ALL_OBJECTS = {"adam": Adam, "sgd": SGD}


def get(identifier) -> Optimizer:
    if isinstance(identifier, str):
        obj_class = ALL_OBJECTS.get(identifier)
        if obj_class is not None:
            return obj_class()

    raise ValueError(f"Could not interpret optimizer identifier: {identifier}")
