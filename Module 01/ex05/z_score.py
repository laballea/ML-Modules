import math
import numpy as np



def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    def var(x):
        try:
            var = 0
            sumX = sum(x)
            for el in x:
                var += (el - (1/len(x)*sumX))**2
            return var / len(x)
        except Exception:
            return None
    def std(x):
        try:
            return math.sqrt(var(x))
        except Exception:
            return None
    if not isinstance(x, np.ndarray):
        return None
    mean = float(sum(x) / len(x))
    vari = std(x)
    result = np.array([(i - mean) / vari for i in x])
    return result.reshape(len(result),)
