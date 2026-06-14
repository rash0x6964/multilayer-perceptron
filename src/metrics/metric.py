import numpy as np


def categorical_accuracy(y_true, y_pred):
    if y_pred.shape[1] == 1:
        pred = (y_pred >= 0.5).astype(int).flatten()
        true = y_true.astype(int).flatten()
        return float(np.mean(pred == true))
    return float(
        np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
    )


def precision(y_true, y_pred):
    pred_class = np.argmax(y_pred, axis=1)
    true_class = np.argmax(y_true, axis=1)
    n_classes = y_true.shape[1]
    scores = []
    for c in range(n_classes):
        tp = np.sum((pred_class == c) & (true_class == c))
        fp = np.sum((pred_class == c) & (true_class != c))
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(scores))


def recall(y_true, y_pred):
    pred_class = np.argmax(y_pred, axis=1)
    true_class = np.argmax(y_true, axis=1)
    n_classes = y_true.shape[1]
    scores = []
    for c in range(n_classes):
        tp = np.sum((pred_class == c) & (true_class == c))
        fn = np.sum((pred_class != c) & (true_class == c))
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(scores))
