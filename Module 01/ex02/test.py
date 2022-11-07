import matplotlib.pyplot as plt
import numpy as np
from fit import predict, plot_with_loss, fit_


x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta = np.array([1, 1]).reshape((-1, 1))
# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=500000)
print(theta1)
"""
Output: array([[1.40709365],
               [1.1150909 ]])
"""
# Example 1:
print(predict(x, theta1))
"""
Output:
array([[15.3408728 ],
       [25.38243697],
       [36.59126492],
       [55.95130097],
       [65.53471499]])
"""
plot_with_loss(x.reshape(len(x),), y.reshape(len(y),), theta1.reshape(2,))

x = np.random.rand(10, 1)
y = np.array([1 + 0.8563 * i for i in x])
theta = np.array([1, 1]).reshape((-1, 1))
# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-4, max_iter=1500000)
plot_with_loss(x.reshape(len(x),), y.reshape(len(y),), [1, 0.8563])
plot_with_loss(x.reshape(len(x),), y.reshape(len(y),), theta1)
print([1, 0.8563], theta1)
