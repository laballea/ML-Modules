import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give
    Args:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature va
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if (not isinstance(x, np.ndarray) or (not isinstance(power, int) and not isinstance(power, list))):
            print("Invalid type")
            return None
        if (isinstance(power, list) and len(power) != x.shape[1]):
            return None
        result = x.copy()
        if not isinstance(power, list):
            for po in range(2, power + 1):
                for col in x.T:
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        else:
            for col, power_el in zip(x.T, power):
                for po in range(2, power_el + 1):
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        return np.array(result)
    except Exception as inst:
        print(inst)
        return None
