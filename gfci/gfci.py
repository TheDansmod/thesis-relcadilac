## This script assumes that the user has pip-installed the pytetrad package. Here is how:
## pip install git+https://github.com/cmu-phil/py-tetrad
## basis of script: https://github.com/cmu-phil/py-tetrad/blob/main/pytetrad/run_TetradSearch.py

import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts

def get_data():
    np.random.seed(42)
    dim = 4
    size = 1000
    beta = np.array([[0, 1, 0, 0],
                     [0, 0, -1.5, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]).T
    omega = np.array([[1.2, 0, 0, 0],
                      [0, 1, 0, 0.6],
                      [0, 0, 1, 0],
                      [0, 0.6, 0, 1]])
    true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
    X = np.random.multivariate_normal([0] * dim, true_sigma, size=size)
    X = X - np.mean(X, axis=0)  # centre the data
    df = pd.DataFrame({f'{i}': X[:, i] for i in range(dim)}, dtype=np.float64)
    return df

def gfci_search(X):
    # X is a n x d numpy matrix (d is number of nodes)
    # returns a numpy matrix (d, d) for a PAG where
    # A[i, j] = 0 means no edge between i and j, else there is an edge
    # A[i, j] = 1 means on edge between i and j, j has a circle
    # A[i, j] = 2 means on edge between i and j, j has a head / arrow
    # A[i, j] = 3 means on edge between i and j, j has a tail
    df = pd.DataFrame({f'{i}': X[:, i] for i in range(X.shape[1])}, dtype=np.float64)
    search = ts.TetradSearch(df)

    ## Use a SEM BIC score and Fisher Z Test
    search.use_sem_bic(penalty_discount=2)
    search.use_fisher_z()

    ## Run the search
    search.run_gfci()
    result = search.get_graph_to_matrix().to_numpy()
    return result

if __name__ == '__main__':
    df = get_data()
    gfci_search(df)
