import numpy as np
from tqdm import tqdm


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(theta).reshape(-1, 1)

    def add_intercept(self, x, type):
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

    def simple_gradient(self, x, y):
        if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray)):
            print("Invalid type.")
            return None
        if (len(y) != len(x) or self.theta.shape[0] != x.shape[1] + 1):
            print("Invalid shape.")
            return None
        fct = 1 / len(x)
        x_hat = self.predict_(x)
        x = self.add_intercept(x, 1).T
        return np.array(fct * (x.dot((x_hat - y))))

    def fit_(self, x, y):
        if (not isinstance(y, np.ndarray)):
            return None
        if (not isinstance(x, np.ndarray)):
            return None
        if (not isinstance(self.theta, np.ndarray)):
            return None
        if (not isinstance(self.alpha, float) and 0 > self.alpha < 1):
            return None
        if (not isinstance(self.max_iter, int) and self.max_iter > 0):
            return None
        for _ in tqdm(range(self.max_iter)):
            grdt = self.simple_gradient(x, y)
            self.theta = self.theta - (grdt * self.alpha)
        return self

    def predict_(self, x):
        if (not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray)):
            print("predict_ invalid type.")
            return None
        if (x.size == 0 or self.theta.size == 0):
            print("predict_ empty array.")
            return None
        if (len(self.theta) != x.shape[1] + 1):
            print("predict_ invalid shape.")
            return None
        x = self.add_intercept(x, 1)
        return x.dot(self.theta)

    def loss_(self, y, y_hat):
        if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
            print("loss_ invalid type.")
            return None
        if (len(y) != len(y_hat)):
            print(len(y), len(y_hat))
            print("loss_ invalid shape.")
            return None
        y = y.reshape(len(y),)
        y_hat = y_hat.reshape(len(y_hat),)
        diff = y - y_hat
        fct = (1 / (2 * len(y)))
        return fct * diff.dot(diff.T)

    def predict(self, x, theta):
        if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
            return None
        if (x.size == 0 or theta.size == 0):
            return None
        if (theta.shape[0] != 2):
            return None
        x = self.add_intercept(x, 1)
        return x.dot(theta)
    
    def mse_(self, y, y_hat):
        return self.loss_(y, y_hat) * 2
