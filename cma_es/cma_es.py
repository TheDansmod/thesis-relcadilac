import random
import functools
from concurrent.futures import ProcessPoolExecutor

import cma
import numpy as np
import matplotlib.pyplot as plt

from relcadilac.data_generator import GraphGenerator
import relcadilac.utils as utils

def objective_fn_order_edge_stability(z, d, tril_ind, data, data_cov, vec2admg, gamma, delta):
    D, B = vec2admg(z, d, tril_ind, None)
    bic = utils.get_bic(D, B, data, data_cov)
    p = z[:d]
    pairwise_diff = np.abs(p[:, None] - p[None, :])[tril_ind]
    order_stability = np.sum(np.minimum(pairwise_diff, delta))
    edge_stability = np.sum(np.minimum(np.abs(z[d:]), delta))
    return bic - gamma * (order_stability + edge_stability)

def objective_fn_z_l2_regularization(z, d, tril_ind, data, data_cov, vec2admg, cmaes_lambda):
    D, B = vec2admg(z, d, tril_ind, None)
    bic = utils.get_bic(D, B, data, data_cov)
    return bic + cmaes_lambda * np.linalg.norm(z) ** 2

def cmaes_admg_search(data, data_cov, admg_model, max_fevals=20_000, verbose=3, popsize=100, num_parallel_workers=8, output_folder='runs/cmaes/', cmaes_lambda=1e-4, gamma=1e-4, delta=1.0, obj_fn_type='order_edge_stability'):
    if obj_fn_type not in ['order_edge_stability', 'z_l2_regularization']:
        raise ValueError("`obj_fn_type` must be one of `order_edge_stability` or `z_l2_regularization`")
    n, d = data.shape
    tril_ind = np.tril_indices(d, -1)
    vec2admg = utils.vec_2_bow_free_admg if admg_model == 'bow-free' else utils.vec_2_ancestral_admg
    if obj_fn_type == 'z_l2_regularization':
        fit_func = functools.partial(objective_fn_z_l2_regularization, d=d, tril_ind=tril_ind, data=data, data_cov=data_cov, vec2admg=vec2admg, cmaes_lambda=cmaes_lambda)
    elif obj_fn_type == 'order_edge_stability':
        fit_func = functools.partial(objective_fn_order_edge_stability, d=d, tril_ind=tril_ind, data=data, data_cov=data_cov, vec2admg=vec2admg, gamma=gamma, delta=delta)
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
    pred_D, pred_B = vec2admg(es.result.xbest, d, tril_ind, None)
    pred_pag = utils.convert_admg_to_pag(pred_D, pred_B)
    pred_bic = utils.get_bic(pred_D, pred_B, data, data_cov)
    return pred_D.astype(int), pred_B, pred_pag, pred_bic, None

if __name__ == '__main__':
    seed = random.randint(1, 100)
    generator = GraphGenerator(seed)
    D, B, data, data_cov, bic, pag = generator.get_admg(num_nodes=10, avg_degree=4, frac_directed=0.7, admg_model='ancestral', do_sampling=True, num_samples=2000, require_connected=True)
    print(f'True D:\n{D}\nTrue B:\n{B}\ntrue bic: {bic}'), 
    cmaes_admg_search(data, data_cov)
