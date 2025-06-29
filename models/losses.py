import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mse_derivative(y_true, y_pred):
    return y_pred - y_true


def binary_cross_entropy(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_derivative(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
