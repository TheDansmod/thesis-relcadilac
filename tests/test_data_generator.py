import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from relcadilac.data_generator import GraphGenerator
from relcadilac.utils import get_transitive_closure

def test_run_bow_free(num_nodes, avg_degree, frac_directed, deg_var, num_iter=10000):
    generator = GraphGenerator()  # no seed
    avg_degrees, frac_directeds = [], []
    for it in range(num_iter):
        adj_d, adj_b, X, S, bic = generator.get_admg(
            num_nodes=num_nodes, 
            avg_degree=avg_degree, 
            frac_directed=frac_directed,
            degree_variance=deg_var,
            admg_model='bow-free',
            plot=False,
            do_sampling=False,
        )
        assert adj_d.shape == (num_nodes, num_nodes)
        assert adj_b.shape == (num_nodes, num_nodes)
        assert np.array_equal(adj_b, adj_b.T)  # symmetric
        assert np.array_equal(adj_d * adj_b, np.zeros((num_nodes, num_nodes)))  # bow-free
        num_dir = np.sum(adj_d)
        num_bidir = np.sum(adj_b) // 2
        frac_directeds.append(num_dir / (num_dir + num_bidir))
        avg_degrees.append((num_dir + num_bidir) / num_nodes)
    assert abs(np.mean(avg_degrees) - avg_degree / 2) < 0.3, f'{np.mean(avg_degrees)=} {avg_degree=}'
    assert abs(np.mean(frac_directeds) - frac_directed) < 0.05, f'{np.mean(frac_directeds)=} {frac_directed=}'

def test_run_ancestral(num_iter=1000):
    generator = GraphGenerator()  # no seed
    excess_avg_degrees, excess_frac_directed = [], []
    for it in tqdm(range(num_iter)):
        num_nodes = random.randint(5, 100)
        avg_degree = random.randint(2, num_nodes-1)
        frac_directed = np.random.uniform(low=0.2, high=0.8, size=(1,))[0]
        adj_d, adj_b, X, S, bic = generator.get_admg(
            num_nodes=num_nodes, 
            avg_degree=avg_degree, 
            frac_directed=frac_directed,
            degree_variance=0.0,
            admg_model='ancestral',
            plot=False,
            do_sampling=False,
        )
        assert adj_d.shape == (num_nodes, num_nodes)
        assert adj_b.shape == (num_nodes, num_nodes)
        assert np.array_equal(adj_b, adj_b.T)  # symmetric
        assert np.array_equal(adj_d * adj_b, np.zeros((num_nodes, num_nodes)))  # bow-free
        dag_tc = get_transitive_closure(num_nodes, adj_d)
        assert np.array_equal(adj_b * dag_tc * dag_tc.T, np.zeros((num_nodes, num_nodes))),  \
                f'Directed Edges:\n{adj_d}\nBidirected Edges:\n{adj_b}\nTransitive Closure:\n{dag_tc.T}'  # ancestral
        num_dir = np.sum(adj_d)
        num_bidir = np.sum(adj_b) // 2
        excess_frac_directed.append((num_dir / (num_dir + num_bidir)) - frac_directed)
        excess_avg_degrees.append(((num_dir + num_bidir) / num_nodes) - (avg_degree / 2))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    axes[0].hist(excess_avg_degrees, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Values')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Excess Average Degree')
    axes[1].hist(excess_frac_directed, bins=30, color='Pink', edgecolor='black')
    axes[1].set_xlabel('Values')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Excess Fraction Directed')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_run_ancestral()
