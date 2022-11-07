import numpy as np


def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    diff = y - y_hat
    fct = (1 / (2 * len(y)))
    return float(fct * diff.T.dot(diff))
