from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features


def plot2d_prediction(act_prices, predicted_prices, mse, axes_labels):
    labels = ["Score", "Predicted score"]
    plt.figure()
    plt.scatter(act_prices[0], act_prices[1], marker='o', color="b", label=labels[0])
    plt.plot(predicted_prices[0], predicted_prices[1], marker='.', color="orange", label=labels[0])
    plt.xlabel(axes_labels[0])
    plt.ylabel(axes_labels[1])
    plt.title(mse)
    plt.legend()

def continuous(x):
    return np.array([np.arange(min(vec),max(vec) + 0.01, 0.01) for vec in x]).reshape(-1, 1)

data = pd.read_csv("are_blue_pills_magics.csv")
X = np.array(data[["Micrograms"]])
Y = np.array(data[["Score"]])
continuous_x = continuous([X])
theta1 = np.array([[83], [-7]]).reshape(-1,1)
theta2 = np.array([[77], [-3], [-1]]).reshape(-1,1)
theta3 = np.array([[43], [33], [-11], [1]]).reshape(-1, 1)
theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
theta5 = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
theta6 = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)

# theta 1
x_ = add_polynomial_features(X, 1)
my_lr = MyLR(theta1, alpha = 1e-5, max_iter = 20000).fit_(x_, Y)
x_continuous = add_polynomial_features(continuous_x, 1)
y_hat_continuous = my_lr.predict_(x_continuous)
y_hat = my_lr.predict_(x_)
plot2d_prediction([X, Y], [continuous_x, y_hat_continuous], my_lr.mse_(Y, y_hat), axes_labels=["micrograms", "score"])


# theta 2
x_ = add_polynomial_features(X, 2)
my_lr = MyLR(theta2, alpha = 1e-5, max_iter = 20000).fit_(x_, Y)
x_continuous = add_polynomial_features(continuous_x, 2)
y_hat_continuous = my_lr.predict_(x_continuous)
y_hat = my_lr.predict_(x_)
plot2d_prediction([X, Y], [continuous_x, y_hat_continuous], my_lr.mse_(Y, y_hat), axes_labels=["micrograms", "score"])


# theta 3
x_ = add_polynomial_features(X, 3)
my_lr = MyLR(theta3, alpha = 1e-5, max_iter = 20000).fit_(x_, Y)
x_continuous = add_polynomial_features(continuous_x, 3)
y_hat_continuous = my_lr.predict_(x_continuous)
y_hat = my_lr.predict_(x_)
plot2d_prediction([X, Y], [continuous_x, y_hat_continuous], my_lr.mse_(Y, y_hat), axes_labels=["micrograms", "score"])

# theta 4
x_ = add_polynomial_features(X, 4)
my_lr = MyLR(theta4, alpha = 1e-6, max_iter = 200000).fit_(x_, Y)
x_continuous = add_polynomial_features(continuous_x, 4)
y_hat_continuous = my_lr.predict_(x_continuous)
y_hat = my_lr.predict_(x_)
plot2d_prediction([X, Y], [continuous_x, y_hat_continuous], my_lr.mse_(Y, y_hat), axes_labels=["micrograms", "score"])


# theta 5
x_ = add_polynomial_features(X, 5)
my_lr = MyLR(theta5, alpha = 5e-8, max_iter = 200000).fit_(x_, Y)
x_continuous = add_polynomial_features(continuous_x, 5)
y_hat_continuous = my_lr.predict_(x_continuous)
y_hat = my_lr.predict_(x_)
plot2d_prediction([X, Y], [continuous_x, y_hat_continuous], my_lr.mse_(Y, y_hat), axes_labels=["micrograms", "score"])


# theta 6
x_ = add_polynomial_features(X, 6)
my_lr = MyLR(theta6, alpha = 1e-9, max_iter = 200000).fit_(x_, Y)
x_continuous = add_polynomial_features(continuous_x, 6)
y_hat_continuous = my_lr.predict_(x_continuous)
y_hat = my_lr.predict_(x_)
plot2d_prediction([X, Y], [continuous_x, y_hat_continuous], my_lr.mse_(Y, y_hat), axes_labels=["micrograms", "score"])

plt.show()
