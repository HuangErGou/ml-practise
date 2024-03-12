import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, x, y):
    return np.mean(-y * np.log(sigmoid(x @ theta)) - (1 - y) * np.log(1 - sigmoid(x @ theta)))


def gradient(theta, x, y):
    return (1 / len(x)) * x.T @ (sigmoid(x @ theta) - y)


def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)


def batch_gradient_decent(theta, x, y, epoch, alpha=0.01):
    cost_data = [cost(theta, x, y)]
    _theta = theta.copy()

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, x, y)
        cost_data.append(cost(_theta, x, y))

    return _theta, cost_data
