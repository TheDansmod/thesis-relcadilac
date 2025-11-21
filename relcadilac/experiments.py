import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import pandas as pd

from relcadilac.data_generator import GraphGenerator
from relcadilac.relcadilac import relcadilac as rel_admg
from relcadilac.metrics import get_admg_metrics, get_pag_metrics
from relcadilac.utils import draw_admg, get_ananke_bic
from gfci.gfci import gfci_search
from dcd.admg_discovery import Discovery

def sample_size_variation():
    sample_sizes = [500, 1000, 2000, 4000]
    experiment_data = []
    for sample_size in sample_sizes:
        for i in range(4):
            print(f'sample size = {sample_size}')
            params = {'num_nodes': 10, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 0, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0}
            num_nodes = params['num_nodes']
            avg_degree = params['avg_degree']
            frac_directed = params['frac_directed']
            degree_variance = params['degree_variance']
            admg_model = params['admg_model']
            params['num_samples'] = sample_size
            D, B, X, S, bic, pag = generator.get_admg(num_nodes=num_nodes, avg_degree=avg_degree, frac_directed=frac_directed, degree_variance=degree_variance, admg_model=admg_model, plot=False, do_sampling=True, num_samples=sample_size)
            # relcadilac
            rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 0, 'ent_coef': params['ent_coef']}
            start = time.perf_counter()
            pred_D, pred_B, pred_pag, _ = rel_admg(X, S, admg_model, steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=0)
            params['relcadilac_time_sec'] = time.perf_counter() - start
            params['relcadilac_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
            params['relcadilac_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            print(f'\trelcadilac done')
            # gfci
            start = time.perf_counter()
            pred_pag = gfci_search(X)
            params['gfci_time_sec'] = time.perf_counter() - start
            params['gfci_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            print(f'\tgfci done')
            # dcd
            df_X = pd.DataFrame({f'{i}': X[:, i] for i in range(num_nodes)})
            admg_class = 'bowfree' if admg_model == 'bow-free' else 'ancestral'
            learn = Discovery()  # using all default parameters
            start = time.perf_counter()
            pred_D, pred_B, pred_pag = learn.discover_admg(df_X, admg_class=admg_class, local=False, num_restarts=params['dcd_num_restarts'])
            params['dcd_time_sec'] = time.perf_counter() - start
            params['dcd_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
            params['dcd_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            print(f'\tdcd done')
            experiment_data.append(params)
            print(params)  # so as not to lose data if a run fails
    with open(r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/thesis/runs/run_002.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

def single_test():
    num_nodes = 6
    avg_degree = 2
    frac_directed = 0.6
    degree_variance = 0.1
    admg_model = 'ancestral'
    num_samples = 1000
    draw_folder = r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/thesis/diagrams/"
    D, B, X, S, bic, pag = generator.get_admg(num_nodes=num_nodes, avg_degree=avg_degree, frac_directed=frac_directed, degree_variance=degree_variance, admg_model=admg_model, plot=False, do_sampling=True, num_samples=num_samples)
    ananke_bic = get_ananke_bic(D, B, X)
    draw_admg(D, B, 'true_admg', draw_folder)
    print(f'true_bic: {bic}\nananke_bic: {ananke_bic}')
    start = time.perf_counter()
    pred_D, pred_B, pred_pag, _ = rel_admg(X, S, admg_model, n_envs=8)
    end = time.perf_counter()
    admg_metrics = get_admg_metrics((D, B), (pred_D, pred_B))
    pag_metrics = get_pag_metrics(pag, pred_pag)
    draw_admg(pred_D, pred_B, 'pred_admg', draw_folder)
    pred_ananke_bic = get_ananke_bic(pred_D, pred_B, X)
    print(f'predicted ananke bic: {pred_ananke_bic}')
    print(f'time taken: {(end - start) / 60} mins\n{admg_metrics}\n{pag_metrics}')


if __name__ == '__main__':
    seed = 20
    generator = GraphGenerator(seed)
    sample_size_variation()
