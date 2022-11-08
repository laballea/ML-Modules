from tqdm import tqdm
from ml42.mylinearregression import MyLinearRegression
import numpy as np


class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(theta).reshape(-1, 1)
        self.lambda_ = lambda_

    def get_params_(self):
        """which get the parameters of the estimator,"""
        return self.alpha, self.max_iter, self.theta, self.lambda_

    def set_params_(self, params):
        """which set the parameters of the estimator,"""
        self.alpha, self.max_iter, self.theta, self.lambda_ = params

    def loss_(self, y, y_hat):
        """which return the loss between 2 vectors (numpy arrays),"""
        try:
            return sum(self.loss_elem_(y, y_hat))
        except Exception as inst:
            print(inst)
            return None

    def loss_elem_(self, y, y_hat):
        """ which return a vector corresponding to the squared diffrence between
        2 vectors (numpy arrays),"""
        try:
            return (y - y_hat) ** 2
        except Exception as inst:
            print(inst)
            return None

    def reg_loss_(self, y, y_hat):
        try:
            theta_cp = np.reshape(self.theta, (len(self.theta), ))
            l2 = float(theta_cp[1:].T.dot(theta_cp[1:]))
            m, n = y.shape
            diff = y_hat - y
            return float((1 / (2 * m)) * (diff.T.dot(diff) + self.lambda_ * l2))
        except Exception as inst:
            print(inst)
            return None

    def gradient_(self, x, y):
        """ which calculates the vectorized regularized gradient,"""
        try:
            m, n = x.shape
            y_hat = self.predict_(x, self.theta)
            x = np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
            diff = y_hat - y
            theta_ = self.theta.copy()
            theta_[0] = 0
            return (1 / m) * (x.T.dot(diff) + (self.lambda_ * theta_))
        except Exception as inst:
            print(inst)
            return None

    def fit_(self, x, y, historic_bl=False):
        """which fits Ridge regression model to a training data"""
        historic = []
        for _ in tqdm(range(self.max_iter), leave=False):
            grdt = self.gradient_(x, y)
            self.theta = self.theta - (grdt * self.alpha)
            if (historic_bl):
                mse = int(self.mse_(y, self.predict_(x)))
                historic.append(mse)
        return historic
