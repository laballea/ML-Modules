from matplotlib import pyplot as plt
import numpy as np


def loss_(y, y_hat):
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    diff = y - y_hat
    fct = (1 / (2 * len(y)))
    return float(fct * diff.T.dot(diff))


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
