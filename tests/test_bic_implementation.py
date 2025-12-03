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
    data_store = {'impl': [], 'num_nodes': [], 'avg_degree': [], 'frac_directed': [], 'admg_model': [], 'num_samples': [], 'runtime': []}
    for num_nodes in tqdm([5, 10, 15, 20, 25, 30][::-1]):
        for i in tqdm(range(30)):
            avg_degree = random.randint(2, num_nodes-1)
            frac_directed = random.uniform(0.2, 0.7)
            degree_variance = 0.1
            admg_model = random.choice(['bow-free', 'ancestral'])
            num_samples = 2000
            D, B, X, S, bic, pag = generator.get_admg(num_nodes, avg_degree, frac_directed, degree_variance=degree_variance, plot=False, num_samples=num_samples, do_sampling=True, get_pag=False, require_connected=True)

            start = time.perf_counter()
            bic = utils.get_bic(D, B, X, S)
            data_store['runtime'].append(time.perf_counter() - start)
            
            start = time.perf_counter()
            my_bic = utils.get_ananke_bic(D, B, X)
            data_store['runtime'].append(time.perf_counter() - start)

            data_store['impl'].append('me')
            data_store['num_nodes'].append(num_nodes)
            data_store['avg_degree'].append(avg_degree)
            data_store['frac_directed'].append(frac_directed)
            data_store['admg_model'].append(admg_model)
            data_store['num_samples'].append(num_samples)

            data_store['impl'].append('ananke')
            data_store['num_nodes'].append(num_nodes)
            data_store['avg_degree'].append(avg_degree)
            data_store['frac_directed'].append(frac_directed)
            data_store['admg_model'].append(admg_model)
            data_store['num_samples'].append(num_samples)
    df = pd.DataFrame(data_store)
    df.to_csv('runs/bic_test_01.csv')
    # plt.plot(my_time, label='me')
    # plt.plot(ananke_time, label='ananke')
    # plt.show()
    # print(ananke_time)
    # print(my_time)
    # print(sum(ananke_time))
    # print(sum(my_time))
    # print((sum(ananke_time) - sum(my_time)) / sum(ananke_time))

def plot_graph():
    df = pd.read_csv('runs/bic_test_02.csv')[['impl', 'num_nodes', 'runtime']].groupby(['impl', 'num_nodes']).agg(['mean', 'std'])
    print(df)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # for color, algo in zip(['r' 'g'], ['me', 'ananke']):
    # me
    # data = df.xs('me', level='impl').reindex([5, 10, 15, 20, 25, 30])
    # x, y_mean, y_std = data.index, data[('runtime', 'mean')], data[('runtime', 'std')]
    # plt.semilogy(x, y_mean, f'-r', label='me')
    # plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='r', alpha=0.15, linewidth=0)
    # # ananke
    # data = df.xs('ananke', level='impl').reindex([5, 10, 15, 20, 25, 30])
    # x, y_mean, y_std = data.index, data[('runtime', 'mean')], data[('runtime', 'std')]
    # plt.semilogy(x, y_mean, f'-g', label='ananke')
    # plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='g', alpha=0.15, linewidth=0)
    # plt.legend()
    # plt.xlabel('Num nodes')
    # plt.ylabel('Runtime (sec)')
    # plt.show()


if __name__ == '__main__':
    generator = GraphGenerator()
    num_runs = 10
    plot_graph()
