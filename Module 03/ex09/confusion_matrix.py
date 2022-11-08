import numpy as np
import pandas as pd


def confusion_matrix_(y, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    labels: optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    df_option: optional, if set to True the function will return a pandas DataFrame
    instead of a numpy array. (default=False)
    Return:
    The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
    None if any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if (labels is None):
            labels0 = list(dict.fromkeys(np.reshape(y, (len(y),))))
            labels1 = list(dict.fromkeys(np.reshape(y_hat, (len(y_hat),))))
            labels = (labels0 if labels0 > labels1 else labels1)
            labels.sort()
            shape = len(labels)
        else:
            shape = len(labels)
        confusion_mtrx = np.zeros((shape, shape))
        for i in range(len(y)):
            if (y[i] not in labels or y_hat[i] not in labels):
                continue
            if y[i] == y_hat[i]:
                confusion_mtrx[labels.index(y[i])][labels.index(y_hat[i])] += 1
            else:
                confusion_mtrx[labels.index(y[i])][labels.index(y_hat[i])] += 1
        if (not df_option):
            return confusion_mtrx.astype(int)
        else:
            return pd.DataFrame(confusion_mtrx.astype(int), columns=labels, index=labels)
    except Exception as inst:
        print(inst)
        return None
