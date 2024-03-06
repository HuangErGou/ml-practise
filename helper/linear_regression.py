import numpy as np


def cost(theta, x, y):
    m = x.shape[0]
    inner = x @ theta - y
    square_sum = inner.T @ inner
    _cost = square_sum / (2 * m)
    return _cost


def gradient(theta, x, y):
    m = x.shape[0]
    inner = x.T @ (x @ theta - y)
    return inner / m


def batch_gradient_decent(theta, x, y, epoch, alpha=0.01):
    cost_data = [cost(theta, x, y)]
    _theta = theta.copy()

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, x, y)
        cost_data.append(cost(_theta, x, y))

    return _theta, cost_data
