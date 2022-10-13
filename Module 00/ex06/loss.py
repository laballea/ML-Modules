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
    # print(len(x.shape), len(theta.shape), theta.shape[0])
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (x.size == 0 or theta.size == 0):
        return None
    if (theta.shape[0] != 2):
        return None
    x = add_intercept(x)
    return x.dot(theta)

def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (not (isinstance(y, np.ndarray) or isinstance(y_hat, np.ndarray))):
        return None
    if (y.shape != y_hat.shape):
        return None
    return np.array([(j - i)**2 for (i, j) in zip(y, y_hat)])

def loss_(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)):
        return None
    if (y.shape != y_hat.shape):
        return None
    return float((1 / (2 * len(y))) * sum(loss_elem_(y, y_hat)))

x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

# Example 1:
print(loss_elem_(y1, y_hat1))  # Output:  array([[0.], [1], [4], [9], [16]])

# Example 2:
print(loss_(y1, y_hat1)) # Output:  3.0
x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
y_hat2 = predict_(x2, theta2)
y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

# Example 3:
print(loss_(y2, y_hat2)) # Output: 2.142857142857143

# Example 4:
print(loss_(y2, y2))  # Output: 0.0