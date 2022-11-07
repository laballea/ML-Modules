import numpy as np
from data_spliter import data_spliter


x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
# Example 1:
print(data_spliter(x1, y, 0.8))
# Output:
"""(array([ 1, 59, 42, 300]), array([10]), array([0, 0, 1, 0]), array([1]))"""
# Example 2:
data_spliter(x1, y, 0.5)
# Output:
"""(array([59, 10]), array([ 1, 300, 42]), array([0, 1]), array([0, 0, 1]))"""
x2 = np.array([[1, 42],
               [300, 10],
               [59, 1],
               [300, 59],
               [10, 42]])
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
# Example 3:
print(data_spliter(x2, y, 0.8))
# Output:
"""(array([[ 10, 42],
        [300, 59],
        [ 59, 1],
        [300, 10]]),
        array([[ 1, 42]]),
array([0, 1, 0, 1]),
array([0]))"""
# Example 4:
print(data_spliter(x2, y, 0.5))
# Output:
"""(array([[59, 1],
[10, 42]]),
array([[300, 10],
        [300, 59],
        [ 1, 42]]),
array([0, 0]),
array([1, 1, 0]))"""
