import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if (not isinstance(x, np.ndarray)):
            print("Invalid type !")
            return None
        if x.size == 0:
            print("Empty array !")
            return None
        return 1 / (1 + np.exp(-x))
    except:
        return None
