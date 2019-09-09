import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rrcf
import os
from time import time

from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool


def test_func(rrcf_forest, n_train, num_trees, ndf_splitted):
    avg_codisp_splitted = np.zeros(ndf_splitted.shape[0])
    for index, point in enumerate(ndf_splitted):
        # Compute average CoDisp

        for tree in rrcf_forest:
            # Insert the new point into the tree
            tree.insert_point(point, index=n_train)
            # Compute codisp on the new point...
            new_codisp = tree.codisp(n_train)
            # And take the average over all trees
            avg_codisp_splitted[index] += new_codisp / num_trees
            tree.forget_point(index=n_train)

    return avg_codisp_splitted


def main():
    dataset_test = "xplane_1473"
    df_test = pd.read_csv('datasets/{}.csv'.format(dataset_test), delimiter='|')
    ndf_test = df_test.to_numpy()
    ndf_test[...] *= 0.8  # bozma kısmı

    print("Test type: {}".format(ndf_test.dtype))

    dataset_train = "xplane_7540"
    df_train = pd.read_csv('datasets/{}.csv'.format(dataset_train), delimiter='|')
    n_train = 7540
    ndf_train = df_train.to_numpy()
    print("Train type: {}".format(ndf_train.dtype))

    ndf_train = ndf_train[:n_train]
    # ndf_train = ndf_train.astype(np.float16)

    if not os.path.isdir(dataset_test):
        os.mkdir(dataset_test)

    start_plot = time()
    for i, col in enumerate(df_test.columns):
        plt.title(col)
        plt.xlabel('time')
        plt.ylabel('value')
        plt.plot(ndf_test[:, i])
        plt.savefig('{}/{}.png'.format(dataset_test, col))
        plt.clf()
        print("{}:{} plotted".format(i, col))

    end_plot = time()
    # Set forest parameters
    num_trees = 100
    tree_size = 256
    sample_size_range = (n_train // tree_size, tree_size)

    # Construct forest
    start_forest = time()
    forest = []
    while len(forest) < num_trees:
        # Select random subsets of points uniformly
        ixs = np.random.choice(n_train, size=sample_size_range, replace=False)

        # Add sampled trees to forest
        trees = [rrcf.RCTree(ndf_train[ix], index_labels=ix) for ix in ixs]
        forest.extend(trees)

    print("Forest constructed")
    end_forest = time()
    # Create a dict to store anomaly score of each point
    # avg_codisp = np.zeros(ndf_test.shape[0])

    cores = cpu_count()
    # create the multiprocessing pool
    pool = Pool(cores)

    start_test = time()
    ndf_test = ndf_test[(ndf_test.shape[0] % cores):]  # to split equally, remove elements
    ndf_splitted = np.split(ndf_test, cores, axis=0)[1:]
    mapped = pool.map(test_func, forest, n_train, num_trees, ndf_splitted)
    avg_codisp = np.vstack(mapped)
    end_test = time()

    print("Finished...\nPlot:   {}\nForest: {}\n{}Test:   {}")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    plt.title("AVG CoDisp")
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.set_ylim(0, 100)
    plt.xlabel('sec/100')
    plt.ylabel('value')
    plt.plot(avg_codisp)
    plt.savefig('{}/result.png'.format(dataset_test))


if __name__ == '__main__':
    main()
