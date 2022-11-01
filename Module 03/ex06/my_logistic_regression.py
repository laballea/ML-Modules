import numpy as np
from tqdm import tqdm

class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
    
    def intercept_(self, x, type=1):
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


    def sigmoid_(self, x):
        if (not isinstance(x, np.ndarray)):
            print("sigmoid_ Invalid type !")
            return None
        if x.size == 0:
            print("sigmoid_ Empty array !")
            return None
        return np.array(1 / (1 + np.exp(-x))).astype(float)

    def predict_(self, x):
        if (not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray)):
            print("predict_ Invalid type !")
            return None
        if x.size == 0 or self.theta.size == 0:
            print("predict_ Empty array !")
            return None
        if x.shape[1] != self.theta.shape[0] - 1:
            print("predict_ Invalid shape !")
            return None
        x = self.add_intercept(x)
        return self.sigmoid_(x.dot(self.theta))

    def loss_elem_(self, y, y_hat):
        return None

    def loss_(self, y, y_hat):
        if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
            print("loss_ Invalid type !")
            return None
        if y.size == 0 or y_hat.size == 0:
            print("loss_ Empty array !")
            return None
        if y.shape != y_hat.shape:
            print("loss_ Invalid shape !")
            return None
        eps = 1e-15
        m, n = y.shape
        ones = np.ones(y.shape)
        return float(-(1 / m) * (y.T.dot(np.log(y_hat + eps)) + (ones - y).T.dot(np.log(ones - y_hat + eps))))

    def gradient(self, x, y):
        if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray)):
            print("gradient Invalid type !")
            return None
        if y.size == 0 or x.size == 0:
            print("gradient Empty array !")
            return None
        if x.shape[1] != self.theta.shape[0] - 1:
            print("gradient Invalid shape !")
            return None
        m, n = x.shape
        y_hat = self.predict_(x)
        x = self.intercept_(x)
        return (1 / m) * (x.T.dot(y_hat - y))

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
        historic = []
        for _ in tqdm(range(self.max_iter), leave=False):
            grdt = self.gradient(x, y)
            self.theta = self.theta - (grdt * self.alpha)
        return historic