import os  # to prevent thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import functools
from concurrent.futures import ProcessPoolExecutor

import cma
import numpy as np

from relcadilac.data_generator import GraphGenerator
import relcadilac.utils as utils

def objective_fn(z, d, tril_ind, data, data_cov):
    D, B = utils.vec_2_bow_free_admg(z, d, tril_ind, None)
    bic = utils.get_bic(D, B, data, data_cov)
    return bic + 1e-4 * np.linalg.norm(z) ** 2

def cmaes_admg_search_no_logs(data, data_cov):
    n, d = data.shape
    tril_ind = np.tril_indices(d, -1)
    fit_func = functools.partial(objective_fn, d=d, tril_ind=tril_ind, data=data, data_cov=data_cov)
    x0, sigma0 = np.random.randn(d * d), 1.0  # initial solution (isotropic), std dev to sample new solutions
    opts = {'maxfevals': 5_000, 'verbose': 3, 'CMA_diagonal': True, 'popsize': int((4 + 3 * np.log(d * d)) * 4)}
    with EvalParallel2(fit_func, number_of_processes=8) as eval_parallel:
        xbest, es = cma.fmin2(None, x0=x0, sigma0=sigma0, options=opts, parallel_objective=eval_parallel)
    # xbest, es = cma.fmin2(objective_function=objective_fn, x0=x0, sigma0=sigma0, options=opts, args=(d, tril_ind, data, data_cov))
    pred_D, pred_B = utils.vec_2_bow_free_admg(xbest, d, tril_ind, None)
    pred_bic = utils.get_bic(pred_D, pred_B, data, data_cov)
    print(f'Predicted D:\n{pred_D.astype(int)}\nPredicted B:\n{pred_B}\npredicted bic: {pred_bic}'), 
    es.plot()

def cmaes_admg_search(data, data_cov):
    n, d = data.shape
    tril_ind = np.tril_indices(d, -1)
    fit_func = functools.partial(objective_fn, d=d, tril_ind=tril_ind, data=data, data_cov=data_cov)
    x0, sigma0 = np.random.randn(d * d), 1.0  # initial solution (isotropic), std dev to sample new solutions
    opts = cma.CMAOptions()
    opts.set('maxfevals', 5000)  # max number of fn evaluations
    opts.set('verbose', 3)
    opts.set('CMA_diagonal', True)  # always only take diagonal updates - else it gave option to i think set probability or something
    opts.set('popsize', int((4 + 3 * np.log(d * d)) * 4))  # double the default
    es = cma.CMAEvolutionStrategy(x0, sigma0, inopts=opts)
    with ProcessPoolExecutor(max_workers=8) as executor:
        while not es.stop():
            X = es.ask()
            F = list(executor.map(fit_func, X))
            es.tell(X, F)
            es.disp()
            es.logger.add()
    pred_D, pred_B = utils.vec_2_bow_free_admg(es.result.xbest, d, tril_ind, None)
    pred_bic = utils.get_bic(pred_D, pred_B, data, data_cov)
    print(f'Predicted D:\n{pred_D.astype(int)}\nPredicted B:\n{pred_B}\npredicted bic: {pred_bic}'), 
    es.plot()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # seed = 32
    # generator = GraphGenerator(seed)
    # D, B, data, data_cov, bic, pag = generator.get_admg(num_nodes=10, avg_degree=4, frac_directed=0.7, admg_model='ancestral', do_sampling=True, num_samples=2000, require_connected=True)
    # print(f'True D:\n{D}\nTrue B:\n{B}\ntrue bic: {bic}'), 
    # cmaes_admg_search(data, data_cov)
    cma.plot()
    plt.show()
