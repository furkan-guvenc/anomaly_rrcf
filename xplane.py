import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rrcf
import os

dataset_name = "xplane_1473"
df = pd.read_csv('datasets/{}.csv'.format(dataset_name), delimiter='|')
n_train = 1000
ndf = df.to_numpy()
#ndf[n_train:, 0] *= 0.9  # bozma kısmı
ndf_train = ndf[:n_train]


if not os.path.isdir(dataset_name):
    os.mkdir(dataset_name)

for i, col in enumerate(df.columns):
    plt.title(col)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.plot(ndf[:, i])
    plt.savefig('{}/{}.png'.format(dataset_name, col))
    plt.clf()
    print("{}:{} plotted".format(i, col))

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
    trees = [rrcf.RCTree(ndf[ix], index_labels=ix) for ix in ixs]
    forest.extend(trees)

print("Forest constructed")
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
avg_codisp = np.zeros(ndf.shape[0])

for index, point in enumerate(ndf):
    # Compute average CoDisp

    for tree in forest:
        # Insert the new point into the tree
        tree.insert_point(point, index=n_train)
        # Compute codisp on the new point...
        new_codisp = tree.codisp(n_train)
        # And take the average over all trees
        avg_codisp[index] += new_codisp / num_trees
        tree.forget_point(index=n_train)

print("Finished...")
plt.title("AVG CoDisp")
plt.xlabel('time')
plt.ylabel('value')
plt.plot(avg_codisp)
plt.savefig('{}/result.png'.format(dataset_name))
exit()
