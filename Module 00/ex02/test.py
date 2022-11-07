import numpy as np
from prediction import simple_predict

x = np.arange(1, 6)
# Example 1:
theta1 = np.array([5, 0])
print(simple_predict(x, theta1))  # Ouput:  array([5., 5., 5., 5., 5.])
# Do you understand why y_hat contains only 5’s here?
# y = Θ₀ + Θ₁ * x, where Θ₀ = 5 and Θ₁ * x will always be 0 because Θ₁ equal 0
# Example 2:
theta2 = np.array([0, 1])
print(simple_predict(x, theta2))
# Output:  array([1., 2., 3., 4., 5.])
# Do you understand why y_hat == x here?
# y = Θ₀ + Θ₁ * x, where Θ₀ = 0 and Θ₁ * x = Θ₁

# Example 3:
theta3 = np.array([5, 3])
print(simple_predict(x, theta3))  # Output:  array([ 8., 11., 14., 17., 20.])

# Example 4:
theta4 = np.array([-3, 1])
print(simple_predict(x, theta4))  # Output:  array([-2., -1., 0., 1., 2.])
