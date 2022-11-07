import numpy as np
from vec_loss import loss_

X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

# Example 1:
print(loss_(X, Y))  # Output: 2.142857142857143

# Example 2:
print(loss_(X, X))  # Output: 0.0
