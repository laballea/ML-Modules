import numpy as np


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible dimensions.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(proportion, float)):
        print("spliter invalid type")
        return None
    if (x.shape[0] != y.shape[0]):
        print("spliter invalid shape")
        return None
    arr = np.concatenate((x, y), axis=1)
    N = len(y)
    X = arr[:, :x.shape[1]]
    Y = arr[:, x.shape[1]]
    sample = int(proportion * N)
    np.random.shuffle(arr)
    x_train, x_test, y_train, y_test = np.array(X[:sample, :]), np.array(X[sample:, :]), np.array(Y[:sample, ]), np.array(Y[sample:, ])
    return (x_train, x_test, y_train, y_test)
