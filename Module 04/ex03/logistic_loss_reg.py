import numpy as np

def regularization(function):
    def new_function(*args, **kwargs):
        ret = function(*args, **kwargs)
        y, y_hat, theta, lambda_ = args
        m, n = y.shape
        theta = np.reshape(theta, (len(theta), ))
        l2 = float(theta[1:].T.dot(theta[1:]))
        return ret + + ((lambda_ / (2 * m)) * l2)
    return new_function

@regularization
def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for l
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta is empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """
    try:
        eps = 1e-15
        m, n = y.shape
        y_hat = y_hat + eps
        ones = np.ones(y.shape)
        return float(-(1 / m) * (y.T.dot(np.log(y_hat)) + (ones - y).T.dot(np.log(ones - y_hat))))
    except Exception as inst:
        print(inst)
        return None