import numpy as np


class EarlyStopping:
    def __init__(
        self, monitor="val_loss", patience=3, restore_best_weights=True
    ):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights

    def on_train_begin(self, model):
        self.wait = 0
        self.best = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs, model):
        current = logs.get(self.monitor)
        if current is None:
            print(
                f"[EarlyStopping] metric '{self.monitor}' not found in logs."
            )
            return

        if current < self.best:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = [
                    (layer.kernel.copy(), layer.bias.copy())
                    for layer in model.layers
                ]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    for layer, (kernel, bias) in zip(
                        model.layers, self.best_weights
                    ):
                        layer.kernel = kernel.copy()
                        layer.bias = bias.copy()
                print(
                    f"\n[EarlyStopping] Stopped at epoch {epoch + 1}. "
                    f"Best {self.monitor}: {self.best:.4f}"
                )
