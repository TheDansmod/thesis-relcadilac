import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from ananke.graphs.admg import ADMG
from ananke.models.linear_gaussian_sem import LinearGaussianSEM as LGSem

from relcadilac.optim_linear_gaussian_sem import LinearGaussianSEM as myLGSem
from relcadilac.data_generator import GraphGenerator

def test_my_bic(num_runs):
    diffs = []
    for i in tqdm(range(100)):
        num_nodes = random.randint(5, 15)
        avg_degree = random.randint(2, num_nodes-1)
        frac_directed = random.uniform(0.2, 0.7)
        degree_variance = 0.1
        admg_model = random.choice(['bow-free', 'ancestral'])
        num_samples = random.choice([1_000, 2_000, 4_000, 8_000])
        D, B, X, S, bic, pag = generator.get_admg(num_nodes, avg_degree, frac_directed, degree_variance=degree_variance, plot=False, num_samples=num_samples, do_sampling=True, get_pag=False)
        d = D.shape[0]
        df_X = pd.DataFrame({f'{i}': X[:, i] for i in range(d)})
        # from library
        vertices = [f'{i}' for i in range(d)]
        di_edges = [(f'{idx[1]}', f'{idx[0]}') for idx, x in np.ndenumerate(D) if x > 0]
        bi_edges = [(f'{idx[0]}', f'{idx[1]}') for idx, x in np.ndenumerate(np.triu(B, 1)) if x > 0]
        graph = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
        model = LGSem(graph)
        model.fit(df_X)
        bic = model.bic(X)
        # my version
        my_model = myLGSem(D, B, X, S)
        my_model.fit()
        my_bic = my_model.bic()
        diffs.append(my_bic - bic)
    plt.hist(diffs, bins=30)
    plt.xlabel("values")
    plt.ylabel("freq")
    plt.title("My BIC Excess")
    plt.show()

if __name__ == '__main__':
    generator = GraphGenerator(42)
    num_runs = 10
    test_my_bic(num_runs)
