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

    def intercept_(self, x):
        try:
            if (not isinstance(x, np.ndarray)):
                print("intercept_ invalid type")
                return None
            return np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
        except Exception as inst:
            print(inst)
            return None

    def sigmoid_(self, x):
        try:
            if (not isinstance(x, np.ndarray)):
                print("sigmoid_ Invalid type !")
                return None
            if x.size == 0:
                print("sigmoid_ Empty array !")
                return None
            return np.array(1 / (1 + np.exp(-x))).astype(float)
        except Exception as inst:
            print(inst)
            return None

    def predict_(self, x):
        try:
            if (not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray)):
                print("predict_ Invalid type !")
                return None
            if x.size == 0 or self.theta.size == 0:
                print("predict_ Empty array !")
                return None
            if x.shape[1] != self.theta.shape[0] - 1:
                print("predict_ Invalid shape !")
                return None
            x = self.intercept_(x)
            return self.sigmoid_(x.dot(self.theta))
        except Exception as inst:
            print(inst)
            return None

    def loss_elem_(self, y, y_hat):
        try:
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
            ones = np.ones(y.shape)
            return float((y.T.dot(np.log(y_hat + eps)) + (ones - y).T.dot(np.log(ones - y_hat + eps))))
        except Exception as inst:
            print(inst)
            return None

    def loss_(self, y, y_hat):
        try:
            if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
                print("loss_ Invalid type !")
                return None
            if y.size == 0 or y_hat.size == 0:
                print("loss_ Empty array !")
                return None
            if y.shape != y_hat.shape:
                print("loss_ Invalid shape !")
                return None
            m, n = y.shape
            return float(-(1 / m) * self.loss_elem_(y, y_hat))
        except Exception as inst:
            print(inst)
            return None

    def gradient(self, x, y):
        try:
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
        except Exception as inst:
            print(inst)
            return None

    def fit_(self, x, y):
        try:
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
        except Exception as inst:
            print(inst)
            return None       
