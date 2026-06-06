from src.metrics.metric import categorical_accuracy
from src.metrics.metric import precision
from src.metrics.metric import recall

ALL_OBJECTS_DICT = {
    "accuracy": categorical_accuracy,
    "precision": precision,
    "recall": recall,
}


def get_metric(identifier):
    if isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier)
        if obj is not None:
            return obj

    raise ValueError(f"Could not interpret metric identifier: {identifier}")
