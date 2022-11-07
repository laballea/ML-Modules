import numpy as np


def add_intercept(x):
    try:
        if (not isinstance(x, np.ndarray)):
            print("intercept_ invalid type")
            return None
        if (len(x.shape) == 1):
            x = np.reshape(x, (x.shape[0], 1))
        return np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
    except Exception as inst:
        print(inst)
        return None


def predict_(x, theta):
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (x.size == 0 or theta.size == 0):
        return None
    if (theta.shape[0] != 2):
        return None
    x = add_intercept(x)
    return x.dot(theta)


def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (not (isinstance(y, np.ndarray) or isinstance(y_hat, np.ndarray))):
        return None
    if (y.shape != y_hat.shape):
        return None
    return np.array([(j - i)**2 for (i, j) in zip(y, y_hat)])


def loss_(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    return float((1 / (2 * len(y))) * sum(loss_elem_(y, y_hat)))
