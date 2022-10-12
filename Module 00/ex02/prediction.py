import numpy as np

def simple_predict(x, theta):
    """
        Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
    """
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (x.size == 0 or theta.size == 0):
        return None
    if (len(x.shape) != 1 or len(theta.shape) != 1 or theta.shape[0] != 2):
        return None
    y_hat = np.array([theta[0] + theta[1] * el for el in x])
    return y_hat

x = np.arange(1,6)
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