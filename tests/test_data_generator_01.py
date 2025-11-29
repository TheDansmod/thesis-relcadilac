from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from relcadilac.data_generator import GraphGenerator

def test_ancestral(num_graphs = 300):
    generator = GraphGenerator()  # no seed
    num_edges, frac = [], []
    for it in range(num_graphs):
        true_D, true_B, data, data_cov, true_bic, true_pag = generator.get_admg(
            num_nodes=10,
            avg_degree=4,
            frac_directed=0.4,
            degree_variance=0.0,
            admg_model='bow-free',
            plot=False,
            do_sampling=False,
            num_samples=0,
            sampling_params=None,
            get_pag=True,
            require_connected=False,
        )
        num_dir = np.sum(true_D)
        num_bidir = np.sum(true_B) // 2
        total_edges = num_dir + num_bidir
        num_edges.append(total_edges)
        frac.append(num_dir / total_edges)
    print('mean edges = ', np.mean(num_edges))
    print('mean frac = ', np.mean(frac))
    print('std edges = ', np.std(num_edges))
    print('std frac = ', np.std(frac))
    plt.hist(num_edges, bins=30, color='skyblue', edgecolor='black')
    plt.show()
    plt.close()
    plt.hist(frac, bins=30, color='skyblue', edgecolor='black')
    plt.show()
    plt.close()

if __name__ == '__main__':
    test_ancestral()
