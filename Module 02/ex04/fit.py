from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

def add_intercept(x):
    if (not isinstance(x, np.ndarray)):
        return None
    if (len(x.shape) == 1):
        x = np.reshape(x, (x.shape[0], 1))
    shape = x.shape
    x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
    return np.reshape(x, (shape[0], shape[1] + 1))

def predict_(x, theta):
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (x.size == 0 or theta.size == 0):
        return None
    if (len(theta) != x.shape[1] + 1):
        return None
    x = add_intercept(x)
    return x.dot(theta)

def gradient(x, y, theta):
    if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        print("Invalid type.")
        return None
    if (len(y) != len(x) or theta.shape[0] != x.shape[1] + 1):
        print("Invalid shape.")
        return None
    fct = 1 / len(x)
    x_hat = predict_(x, theta)
    x = add_intercept(x).T
    return np.array(fct * (x.dot((x_hat - y))))

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a matrix of dimension m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of dimension m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
    None if there is a matching dimension problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
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
        grdt = gradient(x, y, theta)
        theta = theta - (grdt * alpha)
    return theta

