import numpy as np


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    try:
        theta = np.reshape(theta, (len(theta), ))
        sum_ = 0
        for i in range(1, len(theta)):
            sum_ = sum_ + theta[i] ** 2
        return float(sum_)
    except Exception as inst:
        print(inst)
        return None

def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    try:
        theta = np.reshape(theta, (len(theta), ))
        return float(theta[1:].T.dot(theta[1:]))
    except Exception as inst:
        print(inst)
        return None