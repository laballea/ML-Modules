from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


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


def predict(x, theta):
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (x.size == 0 or theta.size == 0):
        return None
    if (theta.shape[0] != 2):
        return None
    x = add_intercept(x)
    return x.dot(theta)


def simple_gradient(x, y, theta):
    if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (len(x) == 0 or len(y) == 0 or len(theta) == 0):
        return None
    if (y.shape != x.shape or theta.shape != (2, 1)):
        return None
    fct = 1 / len(x)
    x_hat = predict(x, theta)
    x = add_intercept(x).T
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
    if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray)):
        return None
    if (not isinstance(theta, np.ndarray)):
        return None
    if (x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1)):
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
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    diff = y - y_hat
    fct = (1 / (2 * len(y)))
    return float(fct * diff.T.dot(diff))


def plot_with_loss(x, y, theta):
    plt.figure()
    for x1, y1 in zip(x, y):
        plt.plot(x1, y1, marker='o', color="blue")
    yline = theta[0] + x * theta[1]
    plt.plot([x, x], [y, yline], color="red", linestyle='dotted')
    loss = loss_(y, yline)
    plt.title("MSE: {:.3f} / Loss: {:.3f}".format(loss * 2, loss))
    plt.plot(x, yline, '-r')
    plt.show()
