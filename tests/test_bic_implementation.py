import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from relcadilac.data_generator import GraphGenerator
import relcadilac.utils as utils

def test_my_bic():
    my_time, ananke_time = [], []
    for i in tqdm(range(100)):
        num_nodes = random.randint(15, 30)
        avg_degree = random.randint(2, num_nodes-1)
        frac_directed = random.uniform(0.2, 0.7)
        degree_variance = 0.1
        admg_model = random.choice(['bow-free', 'ancestral'])
        num_samples = 10000
        D, B, X, S, bic, pag = generator.get_admg(num_nodes, avg_degree, frac_directed, degree_variance=degree_variance, plot=False, num_samples=num_samples, do_sampling=True, get_pag=False)
        start = time.perf_counter()
        bic = utils.get_bic(D, B, X, S)
        my_time.append(time.perf_counter() - start)
        # start = time.perf_counter()
        # my_bic = utils.get_ananke_bic(D, B, X)
        # ananke_time.append(time.perf_counter() - start)
    plt.plot(my_time, label='me')
    # plt.plot(ananke_time, label='ananke')
    plt.show()
    # print(ananke_time)
    # print(my_time)
    # print(sum(ananke_time))
    # print(sum(my_time))
    # print((sum(ananke_time) - sum(my_time)) / sum(ananke_time))


if __name__ == '__main__':
    generator = GraphGenerator()
    num_runs = 10
    test_my_bic()
