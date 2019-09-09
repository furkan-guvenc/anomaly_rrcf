import numpy as np
import pandas as pd
import rrcf
import scipy.io

import matplotlib.pyplot as plt

mat = scipy.io.loadmat('mammography.mat')
np_mat = np.hstack((mat['X'], mat['y']))
df = pd.DataFrame(np_mat)

df = df.drop(columns=6)
ndf = df.to_numpy()

# Set forest parameters
num_trees = 1000
tree_size = 256
n = df.shape[0]  # (11183, 6)
sample_size_range = (n // tree_size, tree_size)

# Construct forest
forest = []
while len(forest) < num_trees:
    # Select random subsets of points uniformly
    ixs = np.random.choice(n, size=sample_size_range,
                           replace=False)
    trees = list()
    for ix in ixs:
        T = ndf[ix]
        trees.append(rrcf.RCTree(T, index_labels=ix))
    # Add sampled trees to foresndf
    #trees = [rrcf.RCTree(ndf[ix], index_labels=ix)
     #        for ix in ixs]
    forest.extend(trees)

# Compute average CoDisp
avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)
for tree in forest:
    codisp = pd.Series({leaf: tree.codisp(leaf)
                        for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
avg_codisp /= index

fig, ax1 = plt.subplots(figsize=(10, 5))

y = np_mat[:, -1] * 50
y += 200

color = 'tab:red'
ax1.set_ylabel('Data', color=color, size=14)
ax1.plot(y, color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.set_ylim(0, 320)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('CoDisp', color=color, size=14)
ax2.plot(pd.Series(avg_codisp).sort_index(), color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
ax2.grid('off')
ax2.set_ylim(0, 320)
plt.title('Sine wave with injected anomaly (red) and anomaly score (blue)', size=14)

plt.show()
exit()
