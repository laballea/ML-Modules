import numpy as np
import matplotlib.pyplot as plt


ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
theta = [510837, 284966, -16108, -1971]
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()