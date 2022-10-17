from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

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


def predict(x, theta):
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
    x_hat = predict(x, theta)
    x = add_intercept(x, 1).T
    return np.array(fct * (x.dot((x_hat - y))))

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension 2 * 1.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(y, np.ndarray)):
        return None
    if (not isinstance(x, np.ndarray)):
        return None
    if (not isinstance(theta, np.ndarray)):
        return None
    if (not isinstance(alpha, float) and 0 > alpha < 1):
        return None
    if (not isinstance(max_iter, int) and max_iter > 0):
        return None
    for _ in tqdm(range(max_iter)):
        grdt = simple_gradient(x, y, theta)
        theta = theta - (grdt * alpha)
    return theta

def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    diff = y - y_hat
    fct = (1 / (2 * len(y)))
    return fct * diff.dot(diff.T)


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    plt.figure()
    for x1, y1 in zip(x, y):
        plt.plot(x1, y1, marker='o', color="blue")
    yline = theta[0] + x * theta[1]
    plt.plot([x, x], [y, yline], color="red", linestyle='dotted')
    loss = loss_(y, yline)
    plt.title("MSE: {:.3f} / Loss: {:.3f}".format(loss * 2, loss))
    plt.plot(x, yline, '-r')
    plt.show()

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta= np.array([1, 1]).reshape((-1, 1))
# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=500000)
print(theta1)
"""
Output: array([[1.40709365],
               [1.1150909 ]])
"""
# Example 1:
print(predict(x, theta1))
"""
Output:
array([[15.3408728 ],
       [25.38243697],
       [36.59126492],
       [55.95130097],
       [65.53471499]])
"""
plot_with_loss(x.reshape(len(x),), y.reshape(len(y),), theta1.reshape(2,))

x = np.random.rand(10, 1)
y = np.array([1 + 0.8563 * i for i in x])
theta = np.array([1, 1]).reshape((-1, 1))
# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-4, max_iter=1500000)
plot_with_loss(x.reshape(len(x),), y.reshape(len(y),), [1, 0.8563])
