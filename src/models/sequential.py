import losses
import optimizers


class Sequential:
    def __init__(self, layers=None):
        self._layers = []
        self.compiled = False
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)

    def compile(self, optimizer="rmsprop", loss=None, metrics=None):
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)
        self.compiled = True

    def fit(self, x=None, y=None, epochs=1, batch_size=None, validation_data=None):
        if not self.compiled:
            raise ValueError(
                "You must call `compile()` before calling `fit()`."
            )

    def evaluate(x, y):
        pass

    def predict(x):
        pass
