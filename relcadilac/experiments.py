import time

from relcadilac.data_generator import GraphGenerator
from relcadilac.relcadilac import relcadilac
from relcadilac.metrics import get_admg_metrics, get_pag_metrics
from relcadilac.utils import draw_admg

def sample_size_variation():
    sample_sizes = [500, 1000, 2000, 4000]
    for sample_size in sample_sizes:
        num_nodes = 10
        avg_degree = 4
        frac_directed = 0.6
        degree_variance = 0.2
        admg_model = 'ancestral'
        plot = False
        do_sampling = True
        num_samples = sample_size

def single_test():
    num_nodes = 5
    avg_degree = 4
    frac_directed = 0.6
    degree_variance = 0.2
    admg_model = 'ancestral'
    plot = False
    do_sampling = True
    num_samples = 1000
    draw_folder = r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/thesis/diagrams/"
    D, B, X, S, bic, pag = generator.get_admg(num_nodes=num_nodes, avg_degree=avg_degree, frac_directed=frac_directed, degree_variance=degree_variance, admg_model=admg_model, plot=plot, do_sampling=do_sampling, num_samples=num_samples)
    draw_admg(D, B, 'true_admg', draw_folder)
    print(f'true_bic: {bic}')
    start = time.perf_counter()
    pred_D, pred_B, pred_pag = relcadilac(X, S)
    end = time.perf_counter()
    admg_metrics = get_admg_metrics((D, B), (pred_D, pred_B))
    pag_metrics = get_pag_metrics(pag, pred_pag)
    draw_admg(pred_D, pred_B, 'pred_admg', draw_folder)
    print(f'time taken: {(end - start) / 60} mins\n{admg_metrics}\n{pag_metrics}')


if __name__ == '__main__':
    seed = 42
    generator = GraphGenerator(seed)
    single_test()
