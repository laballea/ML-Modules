import numpy as np


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array of dimension m * n.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if (not isinstance(x, np.ndarray)):
            print("intercept_ invalid type")
            return None
        if (len(x.shape) == 1):
            x = np.reshape(x, (x.shape[0], 1))
        return np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
    except Exception as inst:
        print(inst)
        return None