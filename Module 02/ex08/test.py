import numpy as np
import matplotlib.pyplot as plt


x = np.arange(1,11).reshape(-1,1)
y = np.array([[ 1.39270298],
[ 3.88237651],
[ 4.37726357],
[ 4.63389049],
[ 7.79814439],
[ 6.41717461],
[ 8.63429886],
[ 8.19939795],
[10.37567392],
[10.68238222]])
plt.scatter(x,y)
plt.show()