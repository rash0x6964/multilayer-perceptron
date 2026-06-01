import numpy as np


class Adam:
    def __init__(
        self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    ):
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0

    def apply_gradients(self, layers):
        self.t += 1
        b1t = 1.0 - self.beta_1**self.t
        b2t = 1.0 - self.beta_2**self.t

        for layer in layers:
            # weights
            layer.mW = self.beta_1 * layer.mW + (1 - self.beta_1) * layer.dW
            layer.vW = self.beta_2 * layer.vW + (1 - self.beta_2) * layer.dW**2
            m_hat_W = layer.mW / b1t
            v_hat_W = layer.vW / b2t
            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)

            # biases
            layer.mb = self.beta_1 * layer.mb + (1 - self.beta_1) * layer.db
            layer.vb = self.beta_2 * layer.vb + (1 - self.beta_2) * layer.db**2
            m_hat_b = layer.mb / b1t
            v_hat_b = layer.vb / b2t
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
