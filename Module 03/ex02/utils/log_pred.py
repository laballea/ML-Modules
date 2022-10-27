import numpy as np

def add_intercept(x, type=1):
    if (not isinstance(x, np.ndarray)):
        return None
    if (len(x.shape) == 1):
        x = np.reshape(x, (x.shape[0], 1))
    shape = x.shape
    if (type == 1):
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
    elif (type == 0):
        x = np.insert(x, 0, np.zeros(x.shape[0]), axis=1)
    return np.reshape(x, (shape[0], shape[1] + 1))

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
    if (not isinstance(x, np.ndarray)):
        print("Invalid type !")
        return None
    if x.size == 0:
        print("Empty array !")
        return None
    return 1 / (1 + np.exp(-x))

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
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        print("Invalid type !")
        return None
    if x.size == 0 or theta.size == 0:
        print("Empty array !")
        return None
    if x.shape[1] != theta.shape[0] - 1:
        print("Invalid shape !")
        return None
    x = add_intercept(x)
    return sigmoid_(x.dot(theta))
    