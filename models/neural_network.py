import numpy as np
from .activations import (
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
)

activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
}


class NeuralNetwork:
    def __init__(self, layers, activation="sigmoid", loss="mse", loss_derivative=None):
        self.layers = layers
        self.activation_name = activation
        self.activation, self.activation_derivative = activation_functions[activation]
        self.loss = loss
        self.loss_derivative = loss_derivative
        self._initialize_weights()

    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            W = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01
            b = np.zeros((1, self.layers[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            a = self.activation(z) if i < len(self.weights) - 1 else z  # output linear
            self.a.append(a)
        return self.a[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        da = self.loss_derivative(y, self.a[-1])
        for i in reversed(range(len(self.weights))):
            dz = da * (
                self.activation_derivative(self.a[i + 1])
                if i != len(self.weights) - 1
                else 1
            )
            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            da = np.dot(dz, self.weights[i].T)

            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=False):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y, y_pred)
            self.backward(X, y, learning_rate)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)
