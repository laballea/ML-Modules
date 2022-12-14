import numpy as np


def intercept_(x):
    """
    add one columns to x
    """
    try:
        if (not isinstance(x, np.ndarray)):
            print("intercept_ invalid type")
            return None
        return np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
    except Exception as inst:
        print(inst)
        return None

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
    except Exception as inst:
        print(inst)
        return None

def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
            print("Invalid type !")
            return None
        if x.size == 0 or theta.size == 0:
            print("Empty array !")
            return None
        if x.shape[1] != theta.shape[0] - 1:
            print("Invalid shape !")
            return None
        x = intercept_(x)
        return sigmoid_(x.dot(theta))
    except Exception as inst:
        print(inst)
        return None
