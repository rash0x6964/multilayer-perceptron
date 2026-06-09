from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def apply_gradients(self, layers):
        pass


class Adam(Optimizer):
    def __init__(
        self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    ):
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self._moments = {}

    def apply_gradients(self, layers):
        self.t += 1

        # bias-correction denominators
        b1t = 1.0 - self.beta_1**self.t
        b2t = 1.0 - self.beta_2**self.t

        for layer in layers:
            lid = id(layer)

            if lid not in self._moments:
                self._moments[lid] = (
                    np.zeros_like(layer.kernel),
                    np.zeros_like(layer.kernel),
                    np.zeros_like(layer.bias),
                    np.zeros_like(layer.bias),
                )

            mW, vW, mb, vb = self._moments[lid]

            # weights
            mW = self.beta_1 * mW + (1 - self.beta_1) * layer.dW
            vW = self.beta_2 * vW + (1 - self.beta_2) * layer.dW**2
            m_hat_W = mW / b1t
            v_hat_W = vW / b2t
            layer.kernel -= (
                self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            )

            # biases
            mb = self.beta_1 * mb + (1 - self.beta_1) * layer.db
            vb = self.beta_2 * vb + (1 - self.beta_2) * layer.db**2
            m_hat_b = mb / b1t
            v_hat_b = vb / b2t
            layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

            self._moments[lid] = (mW, vW, mb, vb)


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self._velocities = {}

    def apply_gradients(self, layers):
        for layer in layers:
            lid = id(layer)

            if lid not in self._velocities:
                self._velocities[lid] = (
                    np.zeros_like(layer.kernel),
                    np.zeros_like(layer.bias),
                )

            vW, vb = self._velocities[lid]

            vW = self.momentum * vW - self.lr * layer.dW
            vb = self.momentum * vb - self.lr * layer.db
            layer.kernel += vW
            layer.bias += vb

            self._velocities[lid] = (vW, vb)
