import numpy as np

def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta are empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """
    try:
        theta = np.reshape(theta, (len(theta), ))
        l2 = float(theta[1:].T.dot(theta[1:]))
        m, n = y.shape
        diff = y_hat - y
        return float((1 / (2 * m)) * (diff.T.dot(diff) + lambda_ * l2))
    except Exception as inst:
        print(inst)
        return None