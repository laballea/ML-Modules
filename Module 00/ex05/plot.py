from matplotlib import pyplot as plt
import numpy as np


def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """
    plt.figure()
    for x1, y1 in zip(x, y):
        plt.plot(x1, y1, marker='o', color="blue")
    x = np.linspace(1, 5, 100)
    y = theta[0] + x * theta[1]
    plt.plot(x, y, '-r')
    plt.show()
