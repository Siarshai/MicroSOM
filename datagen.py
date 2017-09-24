import scipy
import numpy as np

def generate_two_simple_blobs(size_of_cluster=50, skew_factor=1.0, scatter_factor=1.0, border_width=2.0):
    X = scipy.random.standard_normal((size_of_cluster, 2)) + border_width
    X = np.concatenate((X, scatter_factor*(scipy.random.standard_normal((int(skew_factor*size_of_cluster), 2)) - border_width)))
    return X


def generate_five_hills_in_ring(size_of_cluster=80):

    x1_covariance_matrix = [[0.25, 0], [0.1, 0.5]]
    x1_mean = (1, 2)
    x2_covariance_matrix = [[0.5, 1.25], [-0.2, -0.25]]
    x2_mean = (0, -2)
    x3_covariance_matrix = [[0.7, 0.15], [0.15, 0.2]]
    x3_mean = (-3, -3)
    x4_covariance_matrix = [[0.5, -4.75], [0.2, -0.25]]
    x4_mean = (-3, 3)
    x5_covariance_matrix = [[0.15, -1.75], [0.1, -0.1]]
    x5_mean = (0, 6)

    X1 = np.random.multivariate_normal(x1_mean, x1_covariance_matrix, (size_of_cluster))
    X2 = np.random.multivariate_normal(x2_mean, x2_covariance_matrix, (size_of_cluster))
    X3 = np.random.multivariate_normal(x3_mean, x3_covariance_matrix, (size_of_cluster))
    X4 = np.random.multivariate_normal(x4_mean, x4_covariance_matrix, (2*size_of_cluster))
    X5 = np.random.multivariate_normal(x5_mean, x5_covariance_matrix, (3*size_of_cluster//2))
    X = np.concatenate((X1, X2, X3, X4, X5))

    return X