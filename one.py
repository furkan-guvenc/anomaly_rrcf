import numpy as np
import rrcf

import pandas as pd
import matplotlib.pyplot as plt

# Generate data
n = 1000
A = 50
center = 100  # sinus grafiğini yukarı kaydırır.
phi = 30  # sinus grafiğini kaydırır
T = 2 * np.pi / 100
t = np.arange(n)
sin = A * np.sin(T * t - phi * T) + center
# sin[235:255] = 80

# Generate test data
n_test = 1000
A = 50
center = 100  # sinus grafiğini yukarı kaydırır.
phi = 30  # sinus grafiğini kaydırır
T = 2 * np.pi / 100
t = np.arange(n)
sin_test = A * np.sin(T * t - phi * T) + center
sin_test[235:255] = 80
sin_test[500:750] = 80

# Set tree parameters
num_trees = 40
shingle_size = 6
tree_size = 256

# Create a forest of empty trees
forest = []
for _ in range(num_trees):
    tree = rrcf.RCTree()
    forest.append(tree)

points = rrcf.shingle(sin, size=shingle_size)
points_test = rrcf.shingle(sin_test, size=shingle_size)

# Create a dict to store anomaly score of each point
avg_codisp = {}

# For each shingle...
for index, point in enumerate(points):
    # For each tree in the forest...
    for tree in forest:
        # If tree is above permitted size...
        if len(tree.leaves) > tree_size:
            # Drop the oldest point (FIFO)
            tree.forget_point(index - tree_size)

            # Drop the newest point (LIFO)
            # tree.forget_point(tree_size - 1)
        # Insert the new point into the tree
        tree.insert_point(point, index=index)
        # Compute codisp on the new point...
        new_codisp = tree.codisp(index)
        # And take the average over all trees
        if not index in avg_codisp:
           avg_codisp[index] = 0
        avg_codisp[index] += new_codisp / num_trees


# For each shingle...
for index, point in enumerate(points_test):
    # For each tree in the forest...
    for tree in forest:
        # Insert the new point into the tree
        tree.insert_point(point, index=n+1)
        # Compute codisp on the new point...
        new_codisp = tree.codisp(n+1)
        # And take the average over all trees
        if not n+index in avg_codisp:
           avg_codisp[n+index] = 0
        avg_codisp[n+index] += new_codisp / num_trees

        tree.forget_point(index=n+1)


fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_ylabel('Data', color=color, size=14)
ax1.plot(np.concatenate((sin,sin_test), axis=None), color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.set_ylim(0, 160)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('CoDisp', color=color, size=14)
ax2.plot(pd.Series(avg_codisp).sort_index(), color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
ax2.grid('off')
ax2.set_ylim(0, 160)
plt.title('Sine wave with injected anomaly (red) and anomaly score (blue)', size=14)

plt.show()

exit()
