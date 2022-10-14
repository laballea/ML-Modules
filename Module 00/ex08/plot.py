
from matplotlib import pyplot as plt
import numpy as np


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
    xline = np.linspace(min(x), max(x), len(x))
    yline = theta[0] + xline * theta[1]
    plt.plot([x, x], [y, yline], color="red", linestyle='dotted')
    loss = loss_(y, yline)
    plt.title("MSE: {:.3f} / Loss: {:.3f}".format(loss * 2, loss))
    plt.plot(xline, yline, '-r')
    plt.show()


x = np.arange(1,6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# Example 1:
theta1= np.array([18,-1])

plot_with_loss(x, y, theta1)
theta2 = np.array([14, 0])
plot_with_loss(x, y, theta2)

theta3 = np.array([12, 0.8])
plot_with_loss(x, y, theta3)