import time
import random

from tqdm import tqdm
import matplotlib.pyplot as plt
from ananke.models.linear_gaussian_sem import LinearGaussianSEM as LGSem

from relcadilac.data_generator import GraphGenerator
from relcadilac.utils import get_ananke_bic
from relcadilac.optim_linear_gaussian_sem import LinearGaussianSEM as myLGSem

def compare(num_runs):
    diffs, my_bic_time, ananke_bic_time, node_sizes = [], [], [], []
    for i in tqdm(range(num_runs)):
        num_nodes = random.randint(5, 10)
        avg_degree = random.randint(2, num_nodes-1)
        frac_directed = random.uniform(0.2, 0.7)
        degree_variance = 0.1
        admg_model = random.choice(['bow-free', 'ancestral'])
        num_samples = random.choice([500, 1_000, 2_000, 4_000, 8_000])
        D, B, X, S, bic, pag, bic_secs = generator.get_admg(num_nodes, avg_degree, frac_directed, degree_variance=degree_variance, plot=False, num_samples=num_samples, do_sampling=True, get_pag=False)
        start = time.perf_counter()
        ananke_bic = get_ananke_bic(D, B, X)
        ananke_bic_time.append(time.perf_counter() - start)
        my_bic_time.append(bic_secs)
        diffs.append(bic - ananke_bic)
        node_sizes.append(num_nodes)
    fix, (ax1, ax2) = plt.subplots(2)
    ax1.hist(diffs, bins=30)
    ax1.set_xlabel("values")
    ax1.set_ylabel("freq")
    ax1.set_title("My BIC Excess")
    ax2.plot(my_bic_time)
    ax2.plot(ananke_bic_time)
    ax2.legend(['my bic time', 'ananke bic time'])
    ax2.set_xlabel("run num")
    ax2.set_ylabel("secs")
    ax3 = ax2.twinx()
    ax3.plot(node_sizes, color='red')
    ax3.set_ylabel("num nodes")
    plt.show()

if __name__ == '__main__':
    generator = GraphGenerator()
    num_runs = 100
    compare(num_runs)
