import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_
# Example 1:
y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

# Accuracy
# your implementation
print(accuracy_score_(y, y_hat))
"""# Output:
0.5"""
# sklearn implementation
print(accuracy_score(y, y_hat))
"""# Output:
0.5"""

# Precision
# your implementation
print(precision_score_(y, y_hat))
"""# Output:
0.4"""
# sklearn implementation
print(precision_score(y, y_hat))
"""# Output:
0.4"""

# your implementation
print(precision_score_(y, y_hat, pos_label=0))
"""# Output:
0.4"""
# sklearn implementation
print(precision_score(y, y_hat, pos_label=0))
"""# Output:
0.4"""

# Recall
# your implementation
print(recall_score_(y, y_hat))
"""# Output:
0.6666666666666666"""
# sklearn implementation
print(recall_score(y, y_hat))
"""# Output:
0.6666666666666666"""

# your implementation
print(recall_score_(y, y_hat, pos_label=0))
"""# Output:
0.6666666666666666"""
# sklearn implementation
print(recall_score(y, y_hat, pos_label=0))
"""# Output:
0.6666666666666666"""


# F1-score
# your implementation
print(f1_score_(y, y_hat))
"""# Output:
0.5"""
# sklearn implementation
print(f1_score(y, y_hat))
"""# Output:
0.5"""

# Example 2:
y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

# Accuracy
# your implementation
print(accuracy_score_(y, y_hat))
"""# Output:
0.625"""
# sklearn implementation
print(accuracy_score(y, y_hat))
"""# Output:
0.625"""

# Precision
# your implementation
print(precision_score_(y, y_hat, pos_label='dog'))
"""# Output:
0.6"""
# sklearn implementation
print(precision_score(y, y_hat, pos_label='dog'))
"""# Output:
0.6"""

# Recall
# your implementation
print(recall_score_(y, y_hat, pos_label='dog'))
"""# Output:
0.75"""
# sklearn implementation
print(recall_score(y, y_hat, pos_label='dog'))
"""# Output:
0.75"""

# F1-score
# your implementation
print(f1_score_(y, y_hat, pos_label='dog'))
"""# Output:
0.6666666666666665"""
# sklearn implementation
print(f1_score(y, y_hat, pos_label='dog'))
"""# Output:
0.6666666666666665"""

# F1-score
# your implementation
print(f1_score_(y, y_hat, pos_label='norminet'))
"""# Output:
0.6666666666666665"""
# sklearn implementation
print(f1_score(y, y_hat, pos_label='norminet'))
"""# Output:
0.6666666666666665"""