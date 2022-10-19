from matplotlib import pyplot as plt
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
import pandas as pd


def plot2d_prediction(act_prices, predicted_prices, mse, axes_labels):
    labels = ["Sell price", "Predicted sell price"]
    plt.figure()
    plt.scatter(act_prices[0], act_prices[1], marker='o', color="b", label=labels[0])
    plt.scatter(predicted_prices[0], predicted_prices[1], marker='.', color="c", label=labels[0])
    plt.xlabel(axes_labels[0])
    plt.ylabel(axes_labels[1])
    plt.title(mse)
    plt.legend()

data = pd.read_csv("spacecraft_data.csv")

# X = np.array(data[["Age"]])
# Y = np.array(data[["Sell_price"]])
# myLR_age = MyLR(theta = [[1000.0], [-1.0]], alpha = 2.5e-4, max_iter = 25000)
# myLR_age.fit_(X[:,0].reshape(-1,1), Y)
# y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
# plot2d_prediction([X, Y], [X, y_pred], myLR_age.mse_(y_pred,Y), axes_labels=["x1: age (years)", "y: sell price (keuros)"])

# X = np.array(data[["Thrust_power"]])
# Y = np.array(data[["Sell_price"]])
# myLR_age = MyLR(theta = [[1], [1]], alpha = 2.5e-6, max_iter = 25000)
# myLR_age.fit_(X[:,0].reshape(-1,1), Y)
# y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
# plot2d_prediction([X, Y], [X, y_pred], myLR_age.mse_(y_pred,Y), axes_labels=["x2: thrust power (10Km/s)", "y: sell price (keuros)"])

# X = np.array(data[["Terameters"]])
# Y = np.array(data[["Sell_price"]])
# myLR_distance = MyLR(theta = [[732.0], [-3.0]], alpha = 2.5e-6, max_iter = 50000)
# myLR_distance.fit_(X[:,0].reshape(-1,1), Y)
# y_pred = myLR_distance.predict_(X[:,0].reshape(-1,1))
# plot2d_prediction([X, Y], [X, y_pred], myLR_distance.mse_(y_pred,Y), axes_labels=["x3: distance totalizer value of spacecraft (Tmeters)", "y: sell price (keuros)"])

Tera = np.array(data[["Terameters"]])
Age = np.array(data[["Age"]])
Thrust = np.array(data[["Thrust_power"]])
X = np.array(data[["Age","Thrust_power","Terameters"]])
Y = np.array(data[["Sell_price"]])
my_lreg = MyLR(theta = [[1], [1], [1], [1]], alpha = 9e-5, max_iter = 600000)
my_lreg.fit_(X, Y)
y_pred = my_lreg.predict_(X)
print(my_lreg.theta)
plot2d_prediction([Tera, Y], [Tera, y_pred], my_lreg.mse_(y_pred,Y), axes_labels=["x3: distance totalizer value of spacecraft (Tmeters)", "y: sell price (keuros)"])
plot2d_prediction([Age, Y], [Age, y_pred], my_lreg.mse_(y_pred,Y), axes_labels=["x1: age (years)", "y: sell price (keuros)"])
plot2d_prediction([Thrust, Y], [Thrust, y_pred], my_lreg.mse_(y_pred,Y), axes_labels=["x2: thrust power (10Km/s)", "y: sell price (keuros)"])

plt.show()
