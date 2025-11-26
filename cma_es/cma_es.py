import random
import functools
from concurrent.futures import ProcessPoolExecutor

import cma
import numpy as np
import matplotlib.pyplot as plt

from relcadilac.data_generator import GraphGenerator
import relcadilac.utils as utils

def objective_fn(z, d, tril_ind, data, data_cov):
    D, B = utils.vec_2_bow_free_admg(z, d, tril_ind, None)
    bic = utils.get_bic(D, B, data, data_cov)
    return bic + 1e-4 * np.linalg.norm(z) ** 2

def cmaes_admg_search(data, data_cov, max_fevals=20_000, verbose=3, popsize=100, num_parallel_workers=8, output_folder='runs/cmaes/'):
    n, d = data.shape
    tril_ind = np.tril_indices(d, -1)
    fit_func = functools.partial(objective_fn, d=d, tril_ind=tril_ind, data=data, data_cov=data_cov)
    x0, sigma0 = np.random.randn(d * d), 1.0  # initial solution (isotropic), std dev to sample new solutions
    opts = cma.CMAOptions()
    opts.set('maxfevals', max_fevals)  # max number of fn evaluations
    opts.set('verbose', verbose)
    opts.set('CMA_diagonal', True)  # always only take diagonal updates - else it gave option to i think set probability or something
    opts.set('popsize', popsize)
    opts.set('verb_filenameprefix', output_folder)
    es = cma.CMAEvolutionStrategy(x0, sigma0, inopts=opts)
    with ProcessPoolExecutor(max_workers=num_parallel_workers) as executor:
        while not es.stop():
            X = es.ask()
            F = list(executor.map(fit_func, X))
            es.tell(X, F)
            es.disp()
            es.logger.add()
    pred_D, pred_B = utils.vec_2_bow_free_admg(es.result.xbest, d, tril_ind, None)
    pred_pag = utils.convert_admg_to_pag(pred_D, pred_B)
    pred_bic = utils.get_bic(pred_D, pred_B, data, data_cov)
    return pred_D.astype(int), pred_B, pred_pag, pred_bic

if __name__ == '__main__':
    seed = random.randint(1, 100)
    generator = GraphGenerator(seed)
    D, B, data, data_cov, bic, pag = generator.get_admg(num_nodes=10, avg_degree=4, frac_directed=0.7, admg_model='ancestral', do_sampling=True, num_samples=2000, require_connected=True)
    print(f'True D:\n{D}\nTrue B:\n{B}\ntrue bic: {bic}'), 
    cmaes_admg_search(data, data_cov)
