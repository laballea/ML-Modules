from matplotlib import pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR
import numpy as np


data = np.genfromtxt('are_blue_pills_magics.csv', delimiter=',')[1:]
lr = MyLR(np.array([[1], [1]]), 5e-4, 50000)
x = data[:, (1)].reshape(-1, 1)
y = data[:, (2)].reshape(-1, 1)
labels = ["Strue(pills)", "Spredict(pills)"]


def plot_with_loss(x, y, theta):
    plt.figure()
    plt.scatter(x, y, marker='o', color="blue", label=labels[0])
    yline = theta[0] + x * theta[1]
    plt.plot(x, yline, linestyle='--', marker='o', color='green', label=labels[1])
    loss = lr.loss_(y, yline)
    plt.title("MSE: {:.3f} / Loss: {:.3f}".format(loss * 2, loss))
    plt.legend()


def print_loss_theta():
    plt.figure()
    for idx, prct in enumerate(range(-15, 20, 5)):
        theta0 = lr.thetas[0][0] * (1 - (prct / 100))
        result = lr.loss_evolve(x, y, theta0)
        plt.plot(result[:, (1)], result[:, (0)], color=str(0.8 - idx/10), label="J(θ₀={:.1f},θ₁)".format(theta0))
    plt.legend()
    plt.xlabel("θ1")
    plt.xlabel("cost function J")


def solve():
    lr.fit_(x, y)
    plot_with_loss(x.reshape(len(x),), y.reshape(len(y),), lr.thetas)
    print_loss_theta()
    plt.show()


solve()
