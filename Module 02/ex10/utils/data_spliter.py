import numpy as np


def data_spliter(x, y, proportion):
    if (not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(proportion, float)):
        print("spliter invalid type")
        return None
    if (x.shape[0] != y.shape[0]):
        print("spliter invalid shape")
        return None
    arr = np.concatenate((x, y), axis=1)
    N = len(y)
    X = arr[:,:x.shape[1]]
    Y = arr[:,x.shape[1]]
    sample = int(proportion*N)
    np.random.shuffle(arr)
    x_train, x_test, y_train, y_test = np.array(X[:sample,:]), np.array(X[sample:, :]), np.array(Y[:sample, ]).reshape(-1, 1), np.array(Y[sample:,]).reshape(-1, 1)
    return (x_train, x_test, y_train, y_test)
