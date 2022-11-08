import numpy as np


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


def predict_(x, theta):
    x = np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
    return sigmoid_(x.dot(theta))


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        m, n = x.shape
        y_hat = predict_(x, theta)
        gradient = []
        for j in range(len(theta)):
            sum_ = 0
            for i in range(m):
                if (j == 0):
                    sum_ = sum_ + (y_hat[i] - y[i])
                else:
                    sum_ = sum_ + ((y_hat[i] - y[i]) * x[i][j - 1])
            if (j == 0):
                gradient.append((1 / m) * sum_)
            else:
                gradient.append((1 / m) * (sum_ + lambda_ * theta[j]))
        return np.array(gradient).reshape(-1, 1)
    except Exception as inst:
        print(inst)
        return None


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        m, n = x.shape
        y_hat = predict_(x, theta)
        x = np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
        diff = y_hat - y
        theta_ = theta.copy()
        theta_[0] = 0
        return (1 / m) * (x.T.dot(diff) + (lambda_ * theta_))
    except Exception as inst:
        print(inst)
        return None
