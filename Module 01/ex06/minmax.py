import math
import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """

    if not isinstance(x, np.ndarray):
        return None
    minX = min(x)
    maxX = max(x)
    result = np.array([(i - minX) / (maxX - minX) for i in x])
    return result.reshape(len(result),)