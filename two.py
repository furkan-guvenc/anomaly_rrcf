import numpy as np
import pandas as pd
import rrcf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set sample parameters
np.random.seed(0)
n_train = 2000
dim = 3

n_test = 250


# Generate data
X = np.zeros((n_train, dim))
X[:1000, 0] = 5
X[1000:2000, 0] = -5
X += 0.01 * np.random.randn(*X.shape)

Y = np.zeros((n_test, dim))
Y[:100, 0] = 5
#Y[1000:2000, 0] = -5
Y += 0.01 * np.random.randn(*Y.shape)

# Set forest parameters
num_trees = 100
tree_size = 256
sample_size_range = (n_train // tree_size, tree_size)

# Construct forest
forest = []
while len(forest) < num_trees:

    # Select random subsets of points uniformly
    ixs = np.random.choice(n_train, size=sample_size_range, replace=False)

    # Add sampled trees to forest
    trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
    forest.extend(trees)
"""
# Compute average CoDisp
avg_codisp = pd.Series(0.0, index=np.arange(n_train))
index = np.zeros(n_train)
for tree in forest:
    codisp = pd.Series({leaf: tree.codisp(leaf) for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
avg_codisp /= index"""

# Create a dict to store anomaly score of each point
avg_codisp = np.zeros(Y.shape[0])

for index, point in enumerate(Y):
    # Compute average CoDisp

    for tree in forest:
        # Insert the new point into the tree
        tree.insert_point(point, index=n_train)
        # Compute codisp on the new point...
        new_codisp = tree.codisp(n_train)
        # And take the average over all trees
        avg_codisp[index] += new_codisp / num_trees
        tree.forget_point(index=n_train)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


limit = 15

Y_anomaly = Y[avg_codisp >= limit]
Y_normal = Y[avg_codisp < limit]

ax.scatter(Y_normal[:, 0], Y_normal[:, 1], Y_normal[:, 2], marker='o')
ax.scatter(Y_anomaly[:, 0], Y_anomaly[:, 1], Y_anomaly[:, 2], marker='^')


#  ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title("Avarage CoDisp above {}".format(limit), size=14)
plt.show()
