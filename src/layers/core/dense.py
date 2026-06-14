from src.layers.layer import Layer
from src import activations


class Dense(Layer):
    def __init__(
        self,
        units,
        activation=None,
        kernel_initializer="HeUniform",
        input_shape=None,
    ):
        super().__init__()
        self.units = units
        self.input_shape = input_shape
        self.activation_name = activation
        self.activation_fn, self.activation_grad = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self._built = False

        if input_shape is not None:
            self.build(input_shape)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
        )
        self.bias = self.add_weight(
            shape=(self.units),
            initializer="Zeros",
        )

        self._built = True

    def forward(self, inputs):
        if not self._built:
            self.build(inputs.shape)

        self.inputs = inputs
        self.Z = inputs @ self.kernel + self.bias
        self.A = self.activation_fn(self.Z)
        return self.A

    def backward(self, dA):
        batch_size = self.inputs.shape[0]

        if self.activation_name in ("softmax", "sigmoid"):
            # gradient already simplified by loss+softmax combo or loss+sigmoid
            dZ = dA
        else:
            dZ = dA * self.activation_grad(self.A)

        self.dW = self.inputs.T @ dZ / batch_size
        self.db = dZ.mean(axis=0)

        dX = dZ @ self.kernel.T
        return dX
