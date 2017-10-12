from math import sqrt
import numpy as np
import scipy

__author__ = 'Siarshai'


def generate_simple_grid_of_neurons(rows=5, columns=5, squared_euclidean=True):

    W = np.zeros(shape=(rows*columns, 2))
    for i in range(rows):
        for j in range(columns):
            W[i*columns + j, 0] = float(i - 1)/rows
            W[i*columns + j, 1] = float(j - 1)/columns

    D = np.zeros(shape=(rows*columns, rows*columns))
    for i in range(rows):
        for j in range(columns):
            for k in range(rows):
                for l in range(columns):
                    D[i*columns + j, k*columns + l] = (i-k)**2 + (j-l)**2 if squared_euclidean else sqrt((i-k)**2 + (j-l)**2)

    grid = []
    for i in range(columns*rows):
        row = i // rows
        if row == (i - 1) // rows:
            grid.append((i, i - 1))
        if i - rows >= 0:
            grid.append((i, i - rows))

    return W, D, grid


def generate_simple_tree_of_neurons(branching_factor=2, height=4):

    import networkx as nx
    G = nx.balanced_tree(branching_factor, height)

    size = len(G.nodes())
    W = scipy.random.standard_normal((size, 2))

    D = np.zeros(shape=(size, size))
    grid = []
    length_dict = nx.all_pairs_shortest_path_length(G)

    #sorting just in case
    sources = sorted(list(length_dict.keys()))
    for source_node in sources:
        dist_vector = length_dict[source_node]
        destinations = sorted(list(dist_vector.keys()))
        for dest_node in destinations:
            dist = dist_vector[dest_node]
            if dist == 1:
                grid.append((source_node, dest_node))
            D[source_node, dest_node] = dist**2

    return W, D, grid