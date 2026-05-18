import losses
import optimizers


class Sequential:
    def __init__(self, layers=None):
        self.layers = layers or []
        self.compiled = False
        self.stop_training = False

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer="adam", loss="categorical_crossentropy", metrics=None):
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)
        self.metrics = metrics or []
        self.compiled = True

    def fit(self, x=None, y=None, epochs=1, batch_size=None, validation_data=None):
        if not self.compiled:
            raise ValueError(
                "You must call `compile()` before calling `fit()`."
            )

    def _forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _backward(self, y_true, y_pred):
        dA = self.loss.gradient(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def evaluate(x, y):
        pass

    def predict(x):
        pass
