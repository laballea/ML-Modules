import math
import numpy as np


def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    eps: has to be a float, epsilon (default=1e-15)
    Returns:
    The logistic loss value as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        print("Invalid type !")
        return None
    if y.size == 0 or y_hat.size == 0:
        print("Empty array !")
        return None
    if y.shape != y_hat.shape:
        print("Invalid shape !")
        return None
    m, n = y.shape
    y_hat = y_hat + eps
    return float((1 / m) * sum([y[i] * math.log(y_hat[i]) + (1 - y[i]) * math.log(1 - y_hat[i]) for i in range(m)]))