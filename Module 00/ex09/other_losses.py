import numpy as np
from cmath import sqrt


def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    diff = y - y_hat
    fct = (1 / (len(y)))
    return fct * diff.dot(diff.T)


def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    return sqrt(mse_(y, y_hat))


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    diff = abs(y - y_hat)
    fct = (1 / (len(y)))
    return fct * sum(diff)


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    diff = y - y_hat
    diff = diff.dot(diff.T)
    diffmean = y - np.mean(y)
    diffmean = diffmean.dot(diffmean.T)
    return 1 - (diff / diffmean)
