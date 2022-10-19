import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR

# data = pd.read_csv("spacecraft_data.csv")
# X = np.array(data[["Age"]])
# Y = np.array(data[["Sell_price"]])
# myLR_age = MyLR(theta = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
# myLR_age.fit_(X[:,0].reshape(-1,1), Y)
# y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
# print(myLR_age.mse_(Y, y_pred))
# #Output: 55736.86719...

data = pd.read_csv("spacecraft_data.csv")
X = np.array(data[["Age","Thrust_power","Terameters"]])
Y = np.array(data[["Sell_price"]])
my_lreg = MyLR(theta = [[1], [1], [1], [1]], alpha = 9e-5, max_iter = 600000)
# Example 0:
y_pred = my_lreg.predict_(X)
print(my_lreg.mse_(Y, y_pred))
# Output: 144044.877...
# Example 1:
my_lreg.fit_(X,Y)
print(my_lreg.theta)
# Output: array([[334.994...],[-22.535...],[5.857...],[-2.586...]])
# Example 2:
y_pred = my_lreg.predict_(X)
print(my_lreg.mse_(Y, y_pred))
# Output: 586.896999...