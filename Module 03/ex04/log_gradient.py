from pickle import NONE
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

def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    eps: epsilon (default=1e-15)
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
    ones = np.ones(y.shape)
    return float(-(1 / m) * (y.T.dot(np.log(y_hat)) + (ones - y).T.dot(np.log(ones - y_hat))))


def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatibl
    Args:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        print("Invalid type !")
        return None
    if y.size == 0 or x.size == 0:
        print("Empty array !")
        return None
    if x.shape[1] != theta.shape[0] - 1:
        print("Invalid shape !")
        return None
    m, n = x.shape
    y_hat = logistic_predict_(x, theta)
    gradient = np.zeros((n + 1, 1))
    gradient[0] = (1 / m) * sum(y_hat - y)
    for j in range(1, n + 1):
        sum_ = 0
        for i in range(m):
            sum_ = sum_ + (y_hat[i] - y[i]) * x[i][j - 1]
        gradient[j] = (1 / m) * sum_
    return gradient
    