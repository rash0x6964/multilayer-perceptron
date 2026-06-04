from src import initializers


class Layer:
    def __init__(self, name=None): ...

    def __call__(self):
        return self.call()

    def call(self):
        raise NotImplementedError

    def add_weight(self, shape=None, initializer=None):
        initializer = initializers.get(initializer)

        if initializer is None:
            return None

        return initializer(shape)
