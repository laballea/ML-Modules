import numpy as np


def add_polynomial_features(x, power_list):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of dimension m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(x, np.ndarray) or not isinstance(power_list, list)):
        print("Invalid type")
        return None
    if (len(power_list) != len(x.T)):
        print("Poly invalid shape !")
        return None
    result = None
    for col, power in zip(x.T, power_list):
        for po in range(1, power + 1):
            if (result is not None):
                result = np.concatenate((result, (col**po).reshape(-1, 1)),axis=1)   
            else:
                result = (col**po).reshape(-1, 1)
    return np.array(result)