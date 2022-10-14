from matplotlib import pyplot as plt
import numpy as np

def add_intercept(x, type):
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


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (x.size == 0 or theta.size == 0):
        return None
    if (theta.shape[0] != 2):
        return None
    x = add_intercept(x, 1)
    return x.dot(theta)

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (y.shape != x.shape or theta.shape != (2, 1)):
        return None
    fct = 1 / len(x)
    x_hat = predict_(x, theta)
    x = add_intercept(x, 1).T
    return np.array(fct * (x.dot((x_hat - y))))

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

# Example 0:
theta1 = np.array([2, 0.7]).reshape((-1, 1))
print(simple_gradient(x, y, theta1))  # Output:  array([[-19.0342574], [-586.66875564]])

# Example 1:
theta2 = np.array([1, -0.4]).reshape((-1, 1))
print(simple_gradient(x, y, theta2))  # Output: array([[-57.86823748], [-2230.12297889]])