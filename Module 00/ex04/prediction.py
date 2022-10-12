import numpy as np

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    def add_intercept(x):
        if (not isinstance(x, np.ndarray)):
            return None
        if (len(x.shape) == 1):
            x = np.reshape(x, (x.shape[0], 1))
        shape = x.shape
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        return np.reshape(x, (shape[0], shape[1] + 1))

    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (x.size == 0 or theta.size == 0):
        return None
    if (len(x.shape) != 1 or len(theta.shape) != 1 or theta.shape[0] != 2):
        return None
    x = add_intercept(x)
    return x.dot(theta)

x = np.arange(1,6)
# Example 1:
theta1 = np.array([5, 0])
print(predict_(x, theta1))  # Ouput:  array([5., 5., 5., 5., 5.])
# Do you understand why y_hat contains only 5’s here?
# y = Θ₀ + Θ₁ * x, where Θ₀ = 5 and Θ₁ * x will always be 0 because Θ₁ equal 0
# Example 2:
theta2 = np.array([0, 1])
print(predict_(x, theta2))
# Output:  array([1., 2., 3., 4., 5.])
# Do you understand why y_hat == x here?
# y = Θ₀ + Θ₁ * x, where Θ₀ = 0 and Θ₁ * x = Θ₁
# Example 3:
theta3 = np.array([5, 3])
print(predict_(x, theta3))  # Output:  array([ 8., 11., 14., 17., 20.])  
# Example 4:
theta4 = np.array([-3, 1])
print(predict_(x, theta4))  # Output:  array([-2., -1., 0., 1., 2.])