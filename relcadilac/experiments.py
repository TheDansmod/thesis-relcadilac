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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from relcadilac.data_generator import GraphGenerator
from relcadilac.relcadilac import relcadilac as rel_admg
from relcadilac.metrics import get_admg_metrics, get_pag_metrics
from relcadilac.utils import draw_admg, get_ananke_bic, plot_rewards, get_thresholded_admg, convert_admg_to_pag
from gfci.gfci import gfci_search
from dcd.admg_discovery import Discovery

def num_nodes_variation():
    #### DANISH: be careful with the threshold
    num_nodes_list = [5, 10, 15, 20, 30]
    experiment_data = []
    for n_nodes in num_nodes_list:
        for i in range(5):
            print(f'num nodes = {n_nodes}')
            params = {'num_nodes': n_nodes, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 2000, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0}
            D, B, X, S, bic, pag = generator.get_admg(
                    num_nodes=params['num_nodes'],
                    avg_degree=params['avg_degree'],
                    frac_directed=params['frac_directed'],
                    degree_variance=params['degree_variance'],
                    admg_model=params['admg_model'],
                    plot=False,
                    do_sampling=True,
                    num_samples=params['num_samples']
                )
            # relcadilac
            rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 0, 'ent_coef': params['ent_coef']}
            start = time.perf_counter()
            pred_D, pred_B, pred_pag, _, _ = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=0)
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
            df_X = pd.DataFrame({f'{i}': X[:, i] for i in range(n_nodes)})
            admg_class = 'bowfree' if params['admg_model'] == 'bow-free' else 'ancestral'
            learn = Discovery()  # using all default parameters
            start = time.perf_counter()
            pred_D, pred_B, pred_pag = learn.discover_admg(df_X, admg_class=admg_class, local=False, num_restarts=params['dcd_num_restarts'])
            params['dcd_time_sec'] = time.perf_counter() - start
            params['dcd_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
            params['dcd_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            print(f'\tdcd done')
            experiment_data.append(params)
            print(params)  # so as not to lose data if a run fails
    with open(r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/thesis/runs/run_003.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

def sample_size_variation():
    sample_sizes = [500, 1000, 2000, 4000]
    experiment_data = []
    for sample_size in sample_sizes:
        for i in range(5):
            print(f'sample size = {sample_sizes}')
            params = {'num_nodes': 10, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': sample_size, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05}
            D, B, X, S, bic, pag = generator.get_admg(
                    num_nodes=params['num_nodes'],
                    avg_degree=params['avg_degree'],
                    frac_directed=params['frac_directed'],
                    degree_variance=params['degree_variance'],
                    admg_model=params['admg_model'],
                    plot=False,
                    do_sampling=True,
                    num_samples=params['num_samples']
                )
            # relcadilac
            rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 0, 'ent_coef': params['ent_coef']}
            start = time.perf_counter()
            pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=0)
            params['relcadilac_time_sec'] = time.perf_counter() - start
            if params['do_thresholding']:
                thresh_D, thresh_B = get_thresholded_admg(pred_D, pred_B, X, S, threshold=params['threshold'])
                thresh_pag = convert_admg_to_pag(thresh_D, thresh_B)
                params['relcadilac_directed_thresh_adj'] = np.array2string(thresh_D)
                params['relcadilac_bidirected_thresh_adj'] = np.array2string(thresh_B)
                params['relcadilac_thresh_admg_metrics'] = get_admg_metrics((D, B), (thresh_D, thresh_B))
                params['relcadilac_thresh_pag_metrics'] = get_pag_metrics(pag, thresh_pag)
            params['relcadilac_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
            params['relcadilac_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            params['ground_truth_bic'] = bic
            params['relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
            params['relcadilac_pred_bic'] = pred_bic
            params['ground_truth_directed_adj'] = np.array2string(D)
            params['ground_truth_bidirected_adj'] = np.array2string(B)
            print(f'\trelcadilac done')
            # gfci
            # start = time.perf_counter()
            # pred_pag = gfci_search(X)
            # params['gfci_time_sec'] = time.perf_counter() - start
            # params['gfci_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            # print(f'\tgfci done')
            # # dcd
            # df_X = pd.DataFrame({f'{i}': X[:, i] for i in range(n_nodes)})
            # admg_class = 'bowfree' if params['admg_model'] == 'bow-free' else 'ancestral'
            # learn = Discovery()  # using all default parameters
            # start = time.perf_counter()
            # pred_D, pred_B, pred_pag = learn.discover_admg(df_X, admg_class=admg_class, local=False, num_restarts=params['dcd_num_restarts'])
            # params['dcd_time_sec'] = time.perf_counter() - start
            # params['dcd_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
            # params['dcd_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            # print(f'\tdcd done')
            experiment_data.append(params)
            print(params)  # so as not to lose data if a run fails
    with open(r"runs/run_009.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

def test_sample_size_with_ananke_bic():
    # I am using the bic computation from the library and am also tracking the difference between the true BIC and the bic of the final graph
    sample_sizes = [500, 600, 700, 1000, 1500, 2000, 2500, 3000]
    experiment_data = []
    for i in range(10):
        for sample_size in sample_sizes:
            params = {'num_nodes': 7, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': sample_size, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0}
            D, B, X, S, bic, pag = generator.get_admg(
                    num_nodes=params['num_nodes'],
                    avg_degree=params['avg_degree'],
                    frac_directed=params['frac_directed'],
                    degree_variance=params['degree_variance'],
                    admg_model=params['admg_model'],
                    plot=False,
                    do_sampling=True,
                    num_samples=params['num_samples']
                )
            # relcadilac
            rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 0, 'ent_coef': params['ent_coef']}
            start = time.perf_counter()
            pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=0)
            params['relcadilac_time_sec'] = time.perf_counter() - start
            params['relcadilac_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
            params['relcadilac_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            params['ground_truth_bic'] = bic
            params['relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
            params['relcadilac_pred_bic'] = pred_bic
            print(params)
            experiment_data.append(params)
    with open(r"runs/run_007.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

def new_single_test():
    params = {'num_nodes': 5, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 500, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0}
    D, B, X, S, bic, pag = generator.get_admg(
            num_nodes=params['num_nodes'],
            avg_degree=params['avg_degree'],
            frac_directed=params['frac_directed'],
            degree_variance=params['degree_variance'],
            admg_model=params['admg_model'],
            plot=False,
            do_sampling=True,
            num_samples=params['num_samples']
        )
    # relcadilac
    rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 0, 'ent_coef': params['ent_coef']}
    start = time.perf_counter()
    pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=0)
    params['relcadilac_time_sec'] = time.perf_counter() - start
    params['relcadilac_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
    params['relcadilac_pag_metrics'] = get_pag_metrics(pag, pred_pag)
    params['ground_truth_bic'] = bic
    params['relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
    params['relcadilac_pred_bic'] = pred_bic
    print(type(pred_bic))
    # print(params)
    with open(r"runs/run_006.json", "w") as f:
        json.dump(params, f, indent=2)

if __name__ == '__main__':
    seed = 32
    generator = GraphGenerator(seed)
    test_sample_size_with_ananke_bic()
