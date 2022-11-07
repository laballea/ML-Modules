import numpy as np
from plot import plot_with_loss

x = np.arange(1, 6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# Example 1:
theta1 = np.array([18, -1])

plot_with_loss(x, y, theta1)
theta2 = np.array([14, 0])
plot_with_loss(x, y, theta2)

theta3 = np.array([12, 0.8])
plot_with_loss(x, y, theta3)
