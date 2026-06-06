from src.layers import Dense
from src import losses, optimizers, metrics
import numpy as np


class Sequential:
    def __init__(self, layers=None):
        self.layers = layers or []
        self.compiled = False
        self.stop_training = False

    def add(self, layer):
        self.layers.append(layer)

    def compile(
        self,
        optimizer="adam",
        loss="categorical_crossentropy",
        _metrics=["accuracy"],
    ):
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)
        self.metrics = _metrics
        self.metrics_fn = {m: metrics.get_metric(m) for m in self.metrics}
        self.compiled = True

    def _forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _backward(self, y_true, y_pred):
        dA = self.loss.gradient(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def _update(self):
        self.optimizer.apply_gradients(self.layers)

    def _evaluate(self, X, y):
        y_pred = self._forward(X)
        results = {"loss": self.loss.forward(y, y_pred)}
        for name, fn in self.metrics_fn.items():
            results[name] = fn(y, y_pred)
        return results

    def predict(self, X, batch_size=32):
        preds = []
        for start in range(0, len(X), batch_size):
            X_batch = X[start : start + batch_size]
            preds.append(self._forward(X_batch))
        return np.vstack(preds)

    def save(self, path):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f"layer_{i}_W"] = layer.kernel
            data[f"layer_{i}_b"] = layer.bias
            data[f"layer_{i}_units"] = np.array(layer.units)
            data[f"layer_{i}_activation"] = np.array(
                layer.activation_name or "linear"
            )
            data[f"layer_{i}_initializer"] = np.array("glorot_uniform")
        data["n_layers"] = np.array(len(self.layers))
        np.savez(path, **data)
        print(
            f"[save] model saved to '{path}.npz'  ({len(self.layers)} layers)"
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        n = int(data["n_layers"])
        layers = []
        for i in range(n):
            units = int(data[f"layer_{i}_units"])
            activation = str(data[f"layer_{i}_activation"])
            W = data[f"layer_{i}_W"]
            b = data[f"layer_{i}_b"]

            layer = Dense(
                units,
                activation=activation if activation != "linear" else None,
            )

            layer.kernel = W
            layer.bias = b
            layer._built = True
            layers.append(layer)

        model = cls(layers)
        print(f"[load] model loaded from '{path}'  ({n} layers)")
        return model

    def fit(
        self,
        X,
        y,
        epochs=1,
        batch_size=32,
        validation_split=0.0,
        callbacks=None,
        verbose=1,
    ):

        if not self.compiled:
            raise RuntimeError("Call compile() before fit().")

        callbacks = callbacks or []
        metric_names = list(self.metrics_fn.keys())
        history = {
            m: []
            for m in ["loss"]
            + metric_names
            + ["val_loss"]
            + [f"val_{m}" for m in metric_names]
        }

        # validation split
        n_val = int(len(X) * validation_split)
        if n_val > 0:
            X_train, y_train = X[:-n_val], y[:-n_val]
            X_val, y_val = X[-n_val:], y[-n_val:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        for cb in callbacks:
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin(self)

        n_samples = len(X_train)
        n_batches = int(np.ceil(n_samples / batch_size))
        self.stop_training = False

        for epoch in range(epochs):
            # shuffle training data each epoch
            idx = np.random.permutation(n_samples)
            X_s, y_s = X_train[idx], y_train[idx]

            # accumulators for every metric
            epoch_loss = 0.0
            epoch_metric_sums = {m: 0.0 for m in metric_names}

            # batch loop
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_samples)
                X_b, y_b = X_s[start:end], y_s[start:end]

                # forward pass
                y_pred = self._forward(X_b)

                # compute loss scalar (for logging only)
                batch_loss = self.loss.forward(y_b, y_pred)

                # backward pass
                self._backward(y_b, y_pred)

                # update W, b
                self._update()

                epoch_loss += batch_loss * len(X_b)
                for m, fn in self.metrics_fn.items():
                    epoch_metric_sums[m] += fn(y_b, y_pred) * len(X_b)

            epoch_loss /= n_samples
            epoch_metrics = {
                m: epoch_metric_sums[m] / n_samples for m in metric_names
            }

            # validation
            if X_val is not None:
                val_results = self._evaluate(X_val, y_val)
                val_loss = val_results["loss"]
                val_metrics = {m: val_results[m] for m in metric_names}
            else:
                val_loss = None
                val_metrics = {m: None for m in metric_names}

            logs = {"loss": epoch_loss, **epoch_metrics}
            if val_loss is not None:
                logs["val_loss"] = val_loss
                for m in metric_names:
                    logs[f"val_{m}"] = val_metrics[m]

            history["loss"].append(epoch_loss)
            for m in metric_names:
                history[m].append(epoch_metrics[m])
            if val_loss is not None:
                history["val_loss"].append(val_loss)
                for m in metric_names:
                    history[f"val_{m}"].append(val_metrics[m])

            # early stopping checks at the end of each epoch
            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(epoch, logs, self)

            # verbose output
            if verbose:
                train_str = f"loss: {epoch_loss:.4f}  " + "  ".join(
                    f"{m}: {epoch_metrics[m]:.4f}" for m in metric_names
                )
                val_str = ""
                if val_loss is not None:
                    val_str = f"  val_loss: {val_loss:.4f}  " + "  ".join(
                        f"val_{m}: {val_metrics[m]:.4f}" for m in metric_names
                    )
                print(f"Epoch {epoch + 1}/{epochs} — {train_str}{val_str}")

            # stop if early stopping fired
            if self.stop_training:
                break

        return history
