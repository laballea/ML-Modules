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
    x = np.linspace(1,5,100)
    y = theta[0] + x * theta[1]
    plt.plot(x, y, '-r')
    plt.show()

x = np.arange(1,6)
y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
# Example 1:
theta1 = np.array([[4.5], [-0.2]])
plot(x, y, theta1)
theta2 = np.array([[-1.5],[2]])
plot(x, y, theta2)
theta3 = np.array([[3],[0.3]])
plot(x, y, theta3)