import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import random
import warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from relcadilac.data_generator import GraphGenerator
from relcadilac.relcadilac import relcadilac as rel_admg
from relcadilac.metrics import get_admg_metrics, get_pag_metrics
from relcadilac.utils import draw_admg, get_ananke_bic, plot_rewards, get_thresholded_admg, convert_admg_to_pag, get_bic, vec_2_bow_free_admg
from gfci.gfci import gfci_search
from dcd.admg_discovery import Discovery

def num_nodes_variation(seed):
    #### DANISH: be careful with the threshold
    print("\n\nRUNNING NUM NODES VARIATION\n\n")
    num_nodes_list = [5, 10, 15, 20, 30]
    experiment_data = []
    for n_nodes in num_nodes_list:
        for i in range(5):
            print(f'num nodes = {n_nodes}')
            params = {'num_nodes': n_nodes, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 2000, 'admg_model': 'bow-free', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05, 'generator_seed': seed}
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
            params['relcadilac_directed_pred_adj'] = np.array2string(pred_D)
            params['relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
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
    with open(r"runs/run_011.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

def sample_size_variation(seed):
    print("\n\nRUNNING SAMPLE SIZE VARIATION\n\n")
    sample_sizes = [500, 1000, 2000, 4000]
    experiment_data = []
    for sample_size in sample_sizes:
        for i in range(5):
            print(f'sample size = {sample_size}')
            params = {'num_nodes': 10, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': sample_size, 'admg_model': 'bow-free', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05, 'generator_seed': seed}
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
            params['relcadilac_directed_pred_adj'] = np.array2string(pred_D)
            params['relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
            print(f'\trelcadilac done')
            # gfci
            start = time.perf_counter()
            pred_pag = gfci_search(X)
            params['gfci_time_sec'] = time.perf_counter() - start
            params['gfci_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            print(f'\tgfci done')
            # dcd
            df_X = pd.DataFrame({f'{i}': X[:, i] for i in range(params['num_nodes'])})
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
    with open(r"runs/run_012.json", "w") as f:
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

def flatten_data(data):
    # data here is a single dictionary
    reject_keys = ['relcadilac_directed_thresh_adj', 'relcadilac_bidirected_thresh_adj', 'relcadilac_avg_rewards', 'ground_truth_directed_adj', 'ground_truth_bidirected_adj', 'relcadilac_directed_pred_adj', 'relcadilac_bidirected_pred_adj']
    # we need to reject some keys since their values are the matrices or lists
    flat_data = {}
    stack = [(data, '')]
    while stack:
        curr_dict, curr_key = stack.pop()
        for k, val in curr_dict.items():
            new_key = f"{curr_key}_{k}" if curr_key else k
            if isinstance(val, dict):
                stack.append((val, new_key))
            elif new_key in reject_keys:
                continue
            else:
                flat_data[new_key] = val
    return flat_data

def get_df_from_runs(data):
    rows = []
    for exp in data:
        rows.append(flatten_data(exp))
    return pd.DataFrame.from_records(rows)

def create_sample_size_plots(file_path='runs/run_012.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = get_df_from_runs(data)
    averaged_rows = []
    sample_sizes = [500, 1000, 2000, 4000]
    for sample_size in sample_sizes:
        averaged_rows.append(df.query(f'num_samples == {sample_size}').mean(axis=0, numeric_only=True))
    df_new = pd.DataFrame(averaged_rows)
    # tpr fdr f1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, df_new['relcadilac_admg_metrics_admg_tpr'], '-r')
    ax.plot(sample_sizes, df_new['relcadilac_admg_metrics_admg_fdr'], '-g')
    ax.plot(sample_sizes, df_new['relcadilac_admg_metrics_admg_f1'], '-b')
    ax.plot(sample_sizes, df_new['dcd_admg_metrics_admg_tpr'], '--r')
    ax.plot(sample_sizes, df_new['dcd_admg_metrics_admg_fdr'], '--g')
    ax.plot(sample_sizes, df_new['dcd_admg_metrics_admg_f1'], '--b')
    legend_elements_colours = [
        Line2D([0], [0], color='red', lw=2, label='tpr'),
        Line2D([0], [0], color='green', lw=2, label='fdr'),
        Line2D([0], [0], color='blue', lw=2, label='f1'),
    ]
    legend_elements_styles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Relcadilac'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='DCD')
    ]
    ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
    ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Value')
    ax.set_title('DCD vs Relcadilac, ADMG Metrics, Bow-free Graphs')
    plt.savefig('diagrams/bowfree_sample_size_dcd_rel_tpr_fdr_f1_admg.png')
    plt.close()
    # skeleton tpr fdr f1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, df_new['relcadilac_admg_metrics_skeleton_tpr'], '-r')
    ax.plot(sample_sizes, df_new['relcadilac_admg_metrics_skeleton_fdr'], '-g')
    ax.plot(sample_sizes, df_new['relcadilac_admg_metrics_skeleton_f1'], '-b')
    ax.plot(sample_sizes, df_new['dcd_admg_metrics_skeleton_tpr'], '--r')
    ax.plot(sample_sizes, df_new['dcd_admg_metrics_skeleton_fdr'], '--g')
    ax.plot(sample_sizes, df_new['dcd_admg_metrics_skeleton_f1'], '--b')
    ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
    ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Value')
    ax.set_title('DCD vs Relcadilac, ADMG Skeleton Metrics, Bowfree Graphs')
    plt.savefig('diagrams/bowfree_sample_size_dcd_rel_tpr_fdr_f1_admg_skeleton.png')
    plt.close()
    # shd runtime
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, df_new['relcadilac_admg_metrics_admg_shd'], '-b')
    ax.plot(sample_sizes, df_new['dcd_admg_metrics_admg_shd'], '--b')
    ax.set_ylabel('Structural Hamming Distance (SHD)')
    ax1 = ax.twinx()
    ax1.plot(sample_sizes, df_new['relcadilac_time_sec'], '-r')
    ax1.plot(sample_sizes, df_new['dcd_time_sec'], '--r')
    ax1.set_ylabel('Runtime (sec)')
    legend_elements_colours = [
        Line2D([0], [0], color='red', lw=2, label='SHD'),
        Line2D([0], [0], color='blue', lw=2, label='Runtime'),
    ]
    ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
    ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
    ax.legend(handles=legend_elements_styles, loc='lower right', title='Algorithm')
    ax.set_xlabel('Number of Samples')
    ax.set_title('DCD vs Relcadilac, ADMG SHD and Runtime, Bow-free Graphs')
    plt.savefig('diagrams/bowfree_sample_size_dcd_rel_shd_runtime.png')
    plt.close()
    # dcd Relcadilac, gfci (pag graphs)
    types = ['skeleton', 'circle', 'head', 'tail']
    legend_elements_colours = [
        Line2D([0], [0], color='red', lw=2, label='tpr'),
        Line2D([0], [0], color='green', lw=2, label='fdr'),
        Line2D([0], [0], color='blue', lw=2, label='f1'),
    ]
    legend_elements_styles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Relcadilac'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='DCD'),
        Line2D([0], [0], color='black', lw=2, linestyle=':', label='GFCI')
    ]
    for tp in types:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sample_sizes, df_new[f'relcadilac_pag_metrics_{tp}_tpr'], '-r')
        ax.plot(sample_sizes, df_new[f'relcadilac_pag_metrics_{tp}_fdr'], '-g')
        ax.plot(sample_sizes, df_new[f'relcadilac_pag_metrics_{tp}_f1'], '-b')
        ax.plot(sample_sizes, df_new[f'dcd_pag_metrics_{tp}_tpr'], '--r')
        ax.plot(sample_sizes, df_new[f'dcd_pag_metrics_{tp}_fdr'], '--g')
        ax.plot(sample_sizes, df_new[f'dcd_pag_metrics_{tp}_f1'], '--b')
        ax.plot(sample_sizes, df_new[f'gfci_pag_metrics_{tp}_tpr'], ':r')
        ax.plot(sample_sizes, df_new[f'gfci_pag_metrics_{tp}_fdr'], ':g')
        ax.plot(sample_sizes, df_new[f'gfci_pag_metrics_{tp}_f1'], ':b')
        ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
        ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Value')
        ax.set_title(f'DCD vs Relcadilac vs GFCI, PAG {tp} Metrics, Bow-free Graphs')
        plt.savefig(f'diagrams/bowfree_sample_size_dcd_rel_gfci_{tp}_pag_metrics.png')
        plt.close()

def create_num_nodes_plot(file_path='runs/run_011_stdout.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = get_df_from_runs(data)
    df.to_csv('runs/run_011.csv')
    averaged_rows = []
    num_nodes_list = [5, 10, 15, 20, 30]
    for num_nodes in num_nodes_list:
        averaged_rows.append(df.query(f'num_nodes == {num_nodes}').mean(axis=0, numeric_only=True))
    df_new = pd.DataFrame(averaged_rows)
    df_new.to_csv('runs/run_011_avg.csv')
    # tpr fdr f1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(num_nodes_list, df_new['relcadilac_admg_metrics_admg_tpr'], '-r')
    ax.plot(num_nodes_list, df_new['relcadilac_admg_metrics_admg_fdr'], '-g')
    ax.plot(num_nodes_list, df_new['relcadilac_admg_metrics_admg_f1'], '-b')
    ax.plot(num_nodes_list, df_new['dcd_admg_metrics_admg_tpr'], '--r')
    ax.plot(num_nodes_list, df_new['dcd_admg_metrics_admg_fdr'], '--g')
    ax.plot(num_nodes_list, df_new['dcd_admg_metrics_admg_f1'], '--b')
    legend_elements_colours = [
        Line2D([0], [0], color='red', lw=2, label='tpr'),
        Line2D([0], [0], color='green', lw=2, label='fdr'),
        Line2D([0], [0], color='blue', lw=2, label='f1'),
    ]
    legend_elements_styles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Relcadilac'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='DCD')
    ]
    ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
    ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Value')
    ax.set_title('DCD vs Relcadilac, ADMG Metrics, Bow-Free Graphs')
    plt.savefig('diagrams/bowfree_dcd_rel_tpr_fdr_f1_admg.png')
    plt.close()
    print('dcd_rel_tpr_fdr_f1_admg plotted')
    # skeleton tpr fdr f1
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(num_nodes_list, df_new['relcadilac_admg_metrics_skeleton_tpr'], '-r')
    ax.plot(num_nodes_list, df_new['relcadilac_admg_metrics_skeleton_fdr'], '-g')
    ax.plot(num_nodes_list, df_new['relcadilac_admg_metrics_skeleton_f1'], '-b')
    ax.plot(num_nodes_list, df_new['dcd_admg_metrics_skeleton_tpr'], '--r')
    ax.plot(num_nodes_list, df_new['dcd_admg_metrics_skeleton_fdr'], '--g')
    ax.plot(num_nodes_list, df_new['dcd_admg_metrics_skeleton_f1'], '--b')
    ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
    ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Value')
    ax.set_title('DCD vs Relcadilac, ADMG Skeleton Metrics, Bow-Free Graphs')
    plt.savefig('diagrams/bowfree_dcd_rel_tpr_fdr_f1_admg_skeleton.png')
    plt.close()
    print('dcd_rel_tpr_fdr_f1_admg_skeleton plotted')
    # shd runtime
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(num_nodes_list, df_new['relcadilac_admg_metrics_admg_shd'], '-b')
    ax.plot(num_nodes_list, df_new['dcd_admg_metrics_admg_shd'], '--b')
    ax.set_ylabel('Structural Hamming Distance (SHD)')
    ax1 = ax.twinx()
    ax1.plot(num_nodes_list, df_new['relcadilac_time_sec'], '-r')
    ax1.plot(num_nodes_list, df_new['dcd_time_sec'], '--r')
    ax1.set_ylabel('Runtime (sec)')
    legend_elements_colours = [
        Line2D([0], [0], color='red', lw=2, label='SHD'),
        Line2D([0], [0], color='blue', lw=2, label='Runtime'),
    ]
    ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
    ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
    ax.legend(handles=legend_elements_styles, loc='lower right', title='Algorithm')
    ax.set_xlabel('Number of Nodes')
    ax.set_title('DCD vs Relcadilac, ADMG SHD and Runtime, Bow-Free Graphs')
    plt.savefig('diagrams/bowfree_dcd_rel_shd_runtime.png')
    plt.close()
    print('dcd_rel_shd_runtime plotted')
    # dcd Relcadilac, gfci (pag graphs)
    types = ['skeleton', 'circle', 'head', 'tail']
    legend_elements_colours = [
        Line2D([0], [0], color='red', lw=2, label='tpr'),
        Line2D([0], [0], color='green', lw=2, label='fdr'),
        Line2D([0], [0], color='blue', lw=2, label='f1'),
    ]
    legend_elements_styles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Relcadilac'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='DCD'),
        Line2D([0], [0], color='black', lw=2, linestyle=':', label='GFCI')
    ]
    for tp in types:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(num_nodes_list, df_new[f'relcadilac_pag_metrics_{tp}_tpr'], '-r')
        ax.plot(num_nodes_list, df_new[f'relcadilac_pag_metrics_{tp}_fdr'], '-g')
        ax.plot(num_nodes_list, df_new[f'relcadilac_pag_metrics_{tp}_f1'], '-b')
        ax.plot(num_nodes_list, df_new[f'dcd_pag_metrics_{tp}_tpr'], '--r')
        ax.plot(num_nodes_list, df_new[f'dcd_pag_metrics_{tp}_fdr'], '--g')
        ax.plot(num_nodes_list, df_new[f'dcd_pag_metrics_{tp}_f1'], '--b')
        ax.plot(num_nodes_list, df_new[f'gfci_pag_metrics_{tp}_tpr'], ':r')
        ax.plot(num_nodes_list, df_new[f'gfci_pag_metrics_{tp}_fdr'], ':g')
        ax.plot(num_nodes_list, df_new[f'gfci_pag_metrics_{tp}_f1'], ':b')
        ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
        ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Value')
        ax.set_title('DCD vs Relcadilac vs GFCI, PAG Metrics, Bow-free Graphs')
        plt.savefig(f'diagrams/bowfree_dcd_rel_gfci_{tp}_pag_metrics.png')
        plt.close()
        print(f'dcd_rel_gfci_{tp}_pag_metrics plotted')

def plot_merged_sample_size_variation_data():
    # i ran two sample size variation experiments - one with relcadilac, gfci and dcd, but without thresholding (run_001)
    # another with just relcadilac but with thresholding (run_010)
    # i need to create plots for the data, but using the dcd and gfci from run_001 and relcadilac from run_010
    with open(r'runs/run_004.json', 'r') as f:
        data = json.load(f)
    df = get_df_from_runs(data)
    averaged_rows = []
    sample_sizes = [500, 1000, 2000, 4000]
    for sample_size in sample_sizes:
        averaged_rows.append(df.query(f'num_samples == {sample_size}').mean(axis=0, numeric_only=True))
    df2 = pd.DataFrame(averaged_rows)  # this is for the gfci and dcd data from run_001
    with open(r'runs/run_010.json', 'r') as f:
        data = json.load(f)
    df = get_df_from_runs(data)
    averaged_rows = []
    for sample_size in sample_sizes:
        averaged_rows.append(df.query(f'num_samples == {sample_size}').mean(axis=0, numeric_only=True))
    df1 = pd.DataFrame(averaged_rows)  # this is for the relcadilac data from run_010
    # tpr fdr f1
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(sample_sizes, df1['relcadilac_thresh_admg_metrics_admg_tpr'], '-r')
    ax.plot(sample_sizes, df1['relcadilac_thresh_admg_metrics_admg_fdr'], '-g')
    ax.plot(sample_sizes, df1['relcadilac_thresh_admg_metrics_admg_f1'], '-b')
    ax.plot(sample_sizes, df2['dcd_admg_metrics_admg_tpr'], '--r')
    ax.plot(sample_sizes, df2['dcd_admg_metrics_admg_fdr'], '--g')
    ax.plot(sample_sizes, df2['dcd_admg_metrics_admg_f1'], '--b')
    legend_elements_colours = [
        Line2D([0], [0], color='red', lw=2, label='tpr'),
        Line2D([0], [0], color='green', lw=2, label='fdr'),
        Line2D([0], [0], color='blue', lw=2, label='f1'),
    ]
    legend_elements_styles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Relcadilac'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='DCD')
    ]
    ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
    ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Value')
    ax.set_title('DCD vs Relcadilac, ADMG Metrics (Thresholded), Ancestral Graphs')
    plt.savefig('diagrams/sample_size_dcd_rel_thresh_tpr_fdr_f1_admg.png')
    plt.close()
    # skeleton tpr fdr f1
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(sample_sizes, df1['relcadilac_thresh_admg_metrics_skeleton_tpr'], '-r')
    ax.plot(sample_sizes, df1['relcadilac_thresh_admg_metrics_skeleton_fdr'], '-g')
    ax.plot(sample_sizes, df1['relcadilac_thresh_admg_metrics_skeleton_f1'], '-b')
    ax.plot(sample_sizes, df2['dcd_admg_metrics_skeleton_tpr'], '--r')
    ax.plot(sample_sizes, df2['dcd_admg_metrics_skeleton_fdr'], '--g')
    ax.plot(sample_sizes, df2['dcd_admg_metrics_skeleton_f1'], '--b')
    ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
    ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Value')
    ax.set_title('DCD vs Relcadilac, ADMG Skeleton Metrics (Thresholded), Ancestral Graphs')
    plt.savefig('diagrams/sample_size_dcd_rel_thresh_tpr_fdr_f1_admg_skeleton.png')
    plt.close()
    # shd runtime
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(sample_sizes, df1['relcadilac_thresh_admg_metrics_admg_shd'], '-b')
    ax.plot(sample_sizes, df2['dcd_admg_metrics_admg_shd'], '--b')
    ax.set_ylabel('Structural Hamming Distance (SHD)')
    ax1 = ax.twinx()
    ax1.plot(sample_sizes, df1['relcadilac_time_sec'], '-r')
    ax1.plot(sample_sizes, df2['dcd_time_sec'], '--r')
    ax1.set_ylabel('Runtime (sec)')
    legend_elements_colours = [
        Line2D([0], [0], color='red', lw=2, label='SHD'),
        Line2D([0], [0], color='blue', lw=2, label='Runtime'),
    ]
    ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
    ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
    ax.legend(handles=legend_elements_styles, loc='lower right', title='Algorithm')
    ax.set_xlabel('Number of Samples')
    ax.set_title('DCD vs Relcadilac, ADMG SHD and Runtime, Ancestral Graphs')
    plt.savefig('diagrams/sample_size_dcd_rel_thresh_shd_runtime.png')
    plt.close()
    # dcd Relcadilac, gfci (pag graphs)
    types = ['skeleton', 'circle', 'head', 'tail']
    legend_elements_colours = [
        Line2D([0], [0], color='red', lw=2, label='tpr'),
        Line2D([0], [0], color='green', lw=2, label='fdr'),
        Line2D([0], [0], color='blue', lw=2, label='f1'),
    ]
    legend_elements_styles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Relcadilac'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='DCD'),
        Line2D([0], [0], color='black', lw=2, linestyle=':', label='GFCI')
    ]
    for tp in types:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(sample_sizes, df1[f'relcadilac_thresh_pag_metrics_{tp}_tpr'], '-r')
        ax.plot(sample_sizes, df1[f'relcadilac_thresh_pag_metrics_{tp}_fdr'], '-g')
        ax.plot(sample_sizes, df1[f'relcadilac_thresh_pag_metrics_{tp}_f1'], '-b')
        ax.plot(sample_sizes, df2[f'dcd_pag_metrics_{tp}_tpr'], '--r')
        ax.plot(sample_sizes, df2[f'dcd_pag_metrics_{tp}_fdr'], '--g')
        ax.plot(sample_sizes, df2[f'dcd_pag_metrics_{tp}_f1'], '--b')
        ax.plot(sample_sizes, df2[f'gfci_pag_metrics_{tp}_tpr'], ':r')
        ax.plot(sample_sizes, df2[f'gfci_pag_metrics_{tp}_fdr'], ':g')
        ax.plot(sample_sizes, df2[f'gfci_pag_metrics_{tp}_f1'], ':b')
        ax.add_artist(ax.legend(handles=legend_elements_colours, loc='upper left', title='Metrics'))
        ax.legend(handles=legend_elements_styles, loc='lower left', title='Algorithm')
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Value')
        ax.set_title('DCD vs Relcadilac vs GFCI, PAG Metrics, Ancestral Graphs')
        plt.savefig(f'diagrams/sample_size_dcd_rel_thresh_gfci_{tp}_pag_metrics.png')
        plt.close()

def check_rewards_convergence_run_006():
    with open(r'runs/run_006.json', 'r') as f:
        data = json.load(f)
    avg_rewards = list(map(float, data['relcadilac_avg_rewards']))
    plt.plot(avg_rewards)
    plt.xlabel("8 Steps = 1 unit")
    plt.ylabel("reward value")
    plt.title("5 node, degree 4, 500 samples, ancestral graph")
    plt.show()

def check_rewards_convergence_run_013():
    with open(r'runs/run_013.json', 'r') as f:
        data = json.load(f)
    avg_rewards = list(map(float, data['relcadilac_avg_rewards']))
    plt.plot(avg_rewards)
    plt.xlabel("8 Steps = 1 unit")
    plt.ylabel("reward value")
    plt.title("15 node, degree 4, 2000 samples, ancestral graph, 4000 steps_per_env")
    plt.show()

def check_rewards_convergence_run_014():
    with open(r'runs/run_014.json', 'r') as f:
        data = json.load(f)
    avg_rewards = list(map(float, data['relcadilac_avg_rewards']))
    plt.plot(avg_rewards)
    plt.xlabel("8 Steps = 1 unit")
    plt.ylabel("reward value")
    plt.title("10 node, degree 4, 4000 samples, ancestral graph, 4000 steps_per_env")
    plt.show()

def check_rewards_convergence_run_011():
    with open(r'runs/run_011_stdout.json', 'r') as f:
        data = json.load(f)
    for num_nodes in [5, 10, 15, 20, 30]:
        exp_num_nodes = None
        for exp in data:
            if exp['num_nodes'] == num_nodes:
                exp_num_nodes = exp
                break
        avg_rewards = list(map(float, exp_num_nodes['relcadilac_avg_rewards']))
        plt.plot(avg_rewards)
        plt.xlabel("8 Steps = 1 unit")
        plt.ylabel("reward value")
        plt.title(f"{num_nodes} node, degree 4, 2000 samples, bowfree graph")
        plt.savefig(f'diagrams/bow_free_reward_convergence_{num_nodes}_nodes.png')

def check_rewards_convergence():
    with open(r'runs/run_007.json', 'r') as f:
        data = json.load(f)
    sample_sizes = [500, 600, 700, 1000, 1500, 2000, 2500, 3000]
    rewards = {ss: None for ss in sample_sizes} # the list (None to start) for each sample size will contain the averaged rewards over 10 runs
    for single_run in data:
        single_run_rewards = np.array(list(map(float, single_run['relcadilac_avg_rewards']))) / 10  # divide by 10 to average out the rewards since there are 10 runs for each sample size
        if rewards[single_run['num_samples']] is None:
            rewards[single_run['num_samples']] = single_run_rewards
        else:
            rewards[single_run['num_samples']] += single_run_rewards
    for ss in sample_sizes:
        plt.plot(rewards[ss], label=f'{ss}')
    plt.xlabel('8 steps = 1 unit')
    plt.ylabel("reward value")
    plt.legend()
    plt.title("7 node, degree 4, ancestral graph")
    plt.savefig("diagrams/rewards_varying_sample_size_ancestral.png")

def check_bic_variance_sample_size():
    # this function I am essentially checking how the true and predicted bic values (or rather the difference between them) vary as the sample size increases
    with open(r'runs/run_007.json', 'r') as f:
        data = json.load(f)
    sample_sizes = [500, 600, 700, 1000, 1500, 2000, 2500, 3000]
    true_bic = {ss: 0 for ss in sample_sizes} # the list for each sample size will contain the 10 ground truth bic values for that sample size
    pred_bic = {ss: 0 for ss in sample_sizes} # the list for each sample size will contain the 10 predicted bic values for that sample size
    for single_run in data:
        true_bic[single_run['num_samples']] += single_run['ground_truth_bic'] / 10  # divide by 10 to average things out since there were 10 runs
        pred_bic[single_run['num_samples']] += single_run['relcadilac_pred_bic'] / 10  # divide by 10 to average things out since there were 10 runs
    true_bic_list = [true_bic[ss] for ss in sample_sizes]
    pred_bic_list = [pred_bic[ss] for ss in sample_sizes]
    frac_diff = [(pred_bic[ss] - true_bic[ss]) / true_bic[ss] for ss in sample_sizes]
    # plt.plot(sample_sizes, true_bic_list, label='true bic')
    # plt.plot(sample_sizes, pred_bic_list, label='pred bic')
    plt.plot(sample_sizes, frac_diff, label='frac diff')
    plt.xlabel("Sample Sizes")
    plt.ylabel("BIC Values")
    plt.legend()
    plt.title("Fractional Excess of predicted BIC values over true BIC values vs Sample Size, 7 Node, Degree 4, Ancestral Graphs")
    plt.savefig("diagrams/fractional_excess_bic_vs_sample_size.png")

def check_bic_variance_num_nodes():
    # this function I am essentially checking how the true and predicted bic values (or rather the difference between them) vary as the sample size increases
    with open(r'runs/run_011_stdout.json', 'r') as f:
        data = json.load(f)
    num_nodes = [5, 10, 15, 20, 30]
    true_bic = {nn: 0 for nn in num_nodes} # the list for each num nodes will contain the 10 ground truth bic values for that sample size
    pred_bic = {nn: 0 for nn in num_nodes} # the list for each num nodes will contain the 10 predicted bic values for that sample size
    for single_run in data:
        true_bic[single_run['num_nodes']] += single_run['ground_truth_bic'] / 5  # divide by 5 to average things out since there were 5 runs
        pred_bic[single_run['num_nodes']] += single_run['relcadilac_pred_bic'] / 5  # divide by 5 to average things out since there were 5 runs
    true_bic_list = [true_bic[nn] for nn in num_nodes]
    pred_bic_list = [pred_bic[nn] for nn in num_nodes]
    frac_diff = [(pred_bic[nn] - true_bic[nn]) / true_bic[nn] for nn in num_nodes]
    # plt.plot(sample_sizes, true_bic_list, label='true bic')
    # plt.plot(sample_sizes, pred_bic_list, label='pred bic')
    plt.plot(num_nodes, frac_diff, label='frac diff')
    plt.xlabel("Sample Sizes")
    plt.ylabel("BIC Values")
    plt.legend()
    plt.title("Fractional Excess of predicted BIC values over true BIC values vs Num Nodes,\n2000 samples, Degree 4, Bow-free Graphs, 5 runs each")
    plt.savefig("diagrams/num_nodes_variation_bowfree_fractional_excess_bic_vs_sample_size.png")
    # plt.show()

def test_longer_timesteps(seed):
    # the 15 (might be arguable), 20 node and 30 node graphs have not converged and would likely do better if they had more steps
    # due to time constraint, we'll just check 15 node graphs
    experiment_data = []
    for steps_per_env in [2000, 3000, 4000]:
        for i in range(3):
            params = {'num_nodes': 15, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 2000, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05, 'generator_seed': seed}
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
            params['relcadilac_directed_pred_adj'] = np.array2string(pred_D)
            params['relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
            print(f'\trelcadilac done')
            # dcd
            df_X = pd.DataFrame({f'{i}': X[:, i] for i in range(n_nodes)})
            admg_class = 'bowfree' if params['admg_model'] == 'bow-free' else 'ancestral'
            learn = Discovery()  # using all default parameters - except for single restart
            start = time.perf_counter()
            pred_D, pred_B, pred_pag = learn.discover_admg(df_X, admg_class=admg_class, local=False, num_restarts=params['dcd_num_restarts'])
            params['dcd_time_sec'] = time.perf_counter() - start
            params['dcd_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
            params['dcd_pag_metrics'] = get_pag_metrics(pag, pred_pag)
            # params['dcd_pred_bic']
            print(f'\tdcd done')
            experiment_data.append(params)
            print(params)  # so as not to lose data if a run fails
    with open(r"runs/run_011.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

def single_test_02(seed):
    print("\n\nSINGLE TEST 02\n\ni am running an ancestral node 15, 2000 samples, 4000 steps_per_env, 4 degree run to see how long it takes and if the non-convergence issue is still present - only comparing dcd and relcadilac")
    params = {'num_nodes': 15, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 2000, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 4000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05, 'generator_seed': seed}
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
    params['relcadilac_directed_pred_adj'] = np.array2string(pred_D)
    params['relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
    print(f'\trelcadilac done')
    # dcd
    df_X = pd.DataFrame({f'{i}': X[:, i] for i in range(params['num_nodes'])})
    admg_class = 'bowfree' if params['admg_model'] == 'bow-free' else 'ancestral'
    learn = Discovery()  # using all default parameters - except for single restart
    start = time.perf_counter()
    pred_D, pred_B, pred_pag = learn.discover_admg(df_X, admg_class=admg_class, local=False, num_restarts=params['dcd_num_restarts'])
    params['dcd_time_sec'] = time.perf_counter() - start
    params['dcd_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
    params['dcd_pag_metrics'] = get_pag_metrics(pag, pred_pag)
    params['dcd_pred_bic'] = get_bic(pred_D, pred_B, X, S)
    print(f'\tdcd done')
    print(params)  # so as not to lose data if a run fails
    with open(r"runs/run_013.json", "w") as f:
        json.dump(params, f, indent=2)

def single_test_03(seed):
    print("\n\nSINGLE TEST 03\n\ni am running an ancestral node 10, 4000 samples, 4000 steps_per_env, 4 degree run to see if it is possible to recover the true graph - and to see if it is possible to get a better bic than dcd\n\n")
    params = {'num_nodes': 10, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 4000, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 8000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05, 'generator_seed': seed, 'explanation': "i am running an ancestral node 10, 4000 samples, 8000 steps_per_env, 4 degree run to see if it is possible to recover the true graph"}
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
    print(f"\n\nGround truth bic {bic}\n\n")
    params['relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
    params['relcadilac_pred_bic'] = pred_bic
    print(f"\n\nRelcadilac pred bic {pred_bic}\n\n")
    params['ground_truth_directed_adj'] = np.array2string(D)
    params['ground_truth_bidirected_adj'] = np.array2string(B)
    params['relcadilac_directed_pred_adj'] = np.array2string(pred_D)
    params['relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
    print(f'\trelcadilac done')
    print(params)  # so as not to lose data if a run fails
    with open(r"runs/run_015.json", "w") as f:
        json.dump(params, f, indent=2)

def update_algo_params(algo, params, D, B, pred_D, pred_B, X, S, bic, pred_bic, pag, pred_pag, avg_rewards=None):
    # algo could be one of relcadilac, gfci, dcd
    # for other algos there won't be a avg_rewards dictionary
    if params['do_thresholding']:
        thresh_D, thresh_B = get_thresholded_admg(pred_D, pred_B, X, S, threshold=params['threshold'])
        thresh_pag = convert_admg_to_pag(thresh_D, thresh_B)
        params[f'{algo}_thresh_admg_metrics'] = get_admg_metrics((D, B), (thresh_D, thresh_B))
        params[f'{algo}_thresh_pag_metrics'] = get_pag_metrics(pag, thresh_pag)
    params[f'{algo}_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
    params[f'{algo}_pag_metrics'] = get_pag_metrics(pag, pred_pag)
    params['ground_truth_bic'] = bic
    params[f'{algo}_pred_bic'] = pred_bic
    print(f"\n\nGround truth bic {bic}\n\n")
    print(f"\n\n{algo} pred bic {pred_bic}\n\n")
    params['ground_truth_directed_adj'] = np.array2string(D)
    params['ground_truth_bidirected_adj'] = np.array2string(B)
    params[f'{algo}_directed_pred_adj'] = np.array2string(pred_D)
    params[f'{algo}_bidirected_pred_adj'] = np.array2string(pred_B)
    if params['do_thresholding']:
        params[f'{algo}_directed_thresh_adj'] = np.array2string(thresh_D)
        params[f'{algo}_bidirected_thresh_adj'] = np.array2string(thresh_B)
    if avg_rewards is not None:
        params[f'{algo}_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
    return params

def single_test_04(seed):
    explanation = "40,000 steps was also not enough. I am going to try 80,000 steps. I think this will be the last jump I make. Continuation from run 22\nExplanation for run 23: Continuation from run 22, 20_000 steps was not sufficient. I will be running for 40_000 steps and see if that has an impact. I will also be raising the LRU cache size. I will also be outputting the step at which the next best BIC was found so that I can see if better BICs are being found even later. For this run I am fixing the seed to be the same as the seed for run 21.\nExplanation for run 22: I have just figured out that the search capability of my implementation (through PPO) is not lacking. The only reason we weren't able to reach the minimum BIC score was that we weren't searching long enough. I have also learnt that using n_steps=16 is harming my setup. So I will be trying 10 node, 4 degree, 2000 samples, with 20_000 steps_per_env to see if the performance improves - can we reach the ground truth BIC score? The ALIAS implementation actually runs for 64 * 20_000 total timesteps just for the DAGs while I will only be running for 8 * 20_000 total timesteps for ADMGS. I will be using ancestral ADMGs."
    print(f" \n\n SINGLE TEST 04 \n\n {explanation}\n\n")
    params = {'num_nodes': 10, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 2000, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 40_000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 1, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05, 'generator_seed': seed, 'explanation': explanation, 'topo_order_known': False, 'use_logits_partition': False, 'get_pag': True, 'require_connected': False}
    D, B, X, S, bic, pag = generator.get_admg(
            num_nodes=params['num_nodes'],
            avg_degree=params['avg_degree'],
            frac_directed=params['frac_directed'],
            degree_variance=params['degree_variance'],
            admg_model=params['admg_model'],
            plot=False,
            do_sampling=True,
            num_samples=params['num_samples'],
            sampling_params={'beta_low': params['beta_low'], 'beta_high': params['beta_high'], 'omega_offdiag_low': params['omega_offdiag_low'], 'omega_offdiag_high': params['omega_offdiag_high'], 'omega_diag_low': params['omega_diag_low'], 'omega_diag_high': params['omega_diag_high'], 'standardize_data': params['standardize_data'], 'center_data': params['center_data']},
            get_pag=params['get_pag'],
            require_connected=params['require_connected'],
        )
    print(f"\n\nGround truth bic {bic}\n\n")
    # relcadilac - logits partition
    rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 0, 'ent_coef': params['ent_coef']}
    start = time.perf_counter()
    pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=1, topo_order=None, use_logits_partition=params['use_logits_partition'])
    params['logits_relcadilac_time_sec'] = time.perf_counter() - start
    params = update_algo_params('relcadilac', params, D, B, pred_D, pred_B, X, S, bic, pred_bic, pag, pred_pag, avg_rewards=avg_rewards)
    with open('runs/run_024.json', 'w') as f:
        json.dump([params], f, indent=2)
    print(f'\trelcadilac done')

def known_causal_ordering(seed):
    # I want to do an experiment where we already know the causal ordering and are just trying to find the matrix values and edge existence
    print("\n\nKNOWN CAUSAL ORDERING\n\nWe know the causal ordering and just want to find the matrices and edge existence, 10 nodes, 4 degree, 2000 samples, ancestral model, 4000 steps_per_env. 3 runs.")
    experiment_data = []
    for i in range(3):
        print(f"\n\n RUN {i}")
        params = {'num_nodes': 10, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 2000, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05, 'generator_seed': seed, 'explanation': "We know the causal ordering and just want to find the matrices and edge existence, 10 nodes, 4 degree, 2000 samples, ancestral model, 4000 steps_per_env. 3 runs.", 'topo_order_known': True}
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
        nx_graph = nx.from_numpy_array(D.T, create_using=nx.DiGraph)
        topo_order = np.array(list(nx.topological_sort(nx_graph)))
        rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 1, 'ent_coef': params['ent_coef']}
        start = time.perf_counter()
        pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=1, topo_order=None)
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
        print(f"\n\nGround truth bic {bic}\n\n")
        params['relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
        params['relcadilac_pred_bic'] = pred_bic
        print(f"\n\nRelcadilac pred bic {pred_bic}\n\n")
        params['ground_truth_directed_adj'] = np.array2string(D)
        params['ground_truth_bidirected_adj'] = np.array2string(B)
        params['relcadilac_directed_pred_adj'] = np.array2string(pred_D)
        params['relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
        print(f'\trelcadilac done 1')
        # relcadilac - without topo order
        start = time.perf_counter()
        pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=0, topo_order=topo_order)
        params['topo_order_relcadilac_time_sec'] = time.perf_counter() - start
        if params['do_thresholding']:
            thresh_D, thresh_B = get_thresholded_admg(pred_D, pred_B, X, S, threshold=params['threshold'])
            thresh_pag = convert_admg_to_pag(thresh_D, thresh_B)
            params['topo_order_relcadilac_directed_thresh_adj'] = np.array2string(thresh_D)
            params['topo_order_relcadilac_bidirected_thresh_adj'] = np.array2string(thresh_B)
            params['topo_order_relcadilac_thresh_admg_metrics'] = get_admg_metrics((D, B), (thresh_D, thresh_B))
            params['topo_order_relcadilac_thresh_pag_metrics'] = get_pag_metrics(pag, thresh_pag)
        params['topo_order_relcadilac_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
        params['topo_order_relcadilac_pag_metrics'] = get_pag_metrics(pag, pred_pag)
        params['topo_order_relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
        params['topo_order_relcadilac_pred_bic'] = pred_bic
        print(f"\n\nTopo Order Relcadilac pred bic {pred_bic}\n\n")
        params['topo_order_relcadilac_directed_pred_adj'] = np.array2string(pred_D)
        params['topo_order_relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
        print(f'\trelcadilac topo order done')
        print(params)  # so as not to lose data if a run fails
        experiment_data.append(params)
    with open(r"runs/run_016.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

def plot_bic_with_and_without_topo_order():
    with open('runs/run_016.json') as f:
        data = json.load(f)
    true_bic, pred_bic, topo_order_pred_bic = [], [], []
    for exp in data:
        # true_bic.append(exp['ground_truth_bic'])
        tb, pb, topb = exp['ground_truth_bic'], exp['relcadilac_pred_bic'], exp['topo_order_relcadilac_pred_bic']
        pred_bic.append((pb - tb) / tb)
        topo_order_pred_bic.append((topb - tb) / tb)
    # plt.plot(true_bic, label='True BIC')
    plt.plot(pred_bic, label='Predicted BIC Fractional Excess (without topo order)')
    plt.plot(topo_order_pred_bic, label='Predicted BIC Fractional Excess (with topo order)')
    plt.xlabel('Run Number')
    plt.ylabel('Fractional Excess BIC\n(pred - true) / true')
    plt.legend()
    plt.title('Performance of Relcadilac with and without Topological Order')
    plt.savefig('diagrams/ancestral_fraction_excess_bic_with_without_topo_order.png')

def compare_logits_vs_heirarchical_vec2_bowfree_admg(seed):
    # I have two vec2bowfree ADMG functions - one where I use logits and the other which is hierarchical - gives preference to directed edges
    print("\n\nLOGITS VS HIERARCHICAL\n\nI have two vec2bowfree ADMG functions - one where I use logits and the other which is hierarchical - gives preference to directed edges. Everything is default otherwise. 5 runs. Bow-free ADMGs only.\n\n")
    experiment_data = []
    for i in range(1):
        print(f"\n\n RUN {i}")
        params = {'num_nodes': 10, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 2000, 'admg_model': 'bow-free', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05, 'generator_seed': seed, 'explanation': "I have two vec2bowfree ADMG functions - one where I use logits and the other which is hierarchical - gives preference to directed edges. Everything is default otherwise. 5 runs", 'topo_order_known': False, 'use_logits_partition': True}
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
        # relcadilac - logits partition
        rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 0, 'ent_coef': params['ent_coef']}
        start = time.perf_counter()
        pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=0, topo_order=None, use_logits_partition=True)
        params['logits_relcadilac_time_sec'] = time.perf_counter() - start
        if params['do_thresholding']:
            thresh_D, thresh_B = get_thresholded_admg(pred_D, pred_B, X, S, threshold=params['threshold'])
            thresh_pag = convert_admg_to_pag(thresh_D, thresh_B)
            params['logits_relcadilac_directed_thresh_adj'] = np.array2string(thresh_D)
            params['logits_relcadilac_bidirected_thresh_adj'] = np.array2string(thresh_B)
            params['logits_relcadilac_thresh_admg_metrics'] = get_admg_metrics((D, B), (thresh_D, thresh_B))
            params['logits_relcadilac_thresh_pag_metrics'] = get_pag_metrics(pag, thresh_pag)
        params['logits_relcadilac_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
        params['logits_relcadilac_pag_metrics'] = get_pag_metrics(pag, pred_pag)
        params['ground_truth_bic'] = bic
        print(f"\n\nGround truth bic {bic}\n\n")
        params['logits_relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
        params['logits_relcadilac_pred_bic'] = pred_bic
        print(f"\n\nLogits Relcadilac pred bic {pred_bic}\n\n")
        params['ground_truth_directed_adj'] = np.array2string(D)
        params['ground_truth_bidirected_adj'] = np.array2string(B)
        params['logits_relcadilac_directed_pred_adj'] = np.array2string(pred_D)
        params['logits_relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
        print(f'\trelcadilac done logits')
        # relcadilac - normal
        start = time.perf_counter()
        pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=0, topo_order=None, use_logits_partition=False)
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
        params['relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
        params['relcadilac_pred_bic'] = pred_bic
        print(f"\n\nRelcadilac pred bic {pred_bic}\n\n")
        params['relcadilac_directed_pred_adj'] = np.array2string(pred_D)
        params['relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
        print(f'\trelcadilac done normal')
        print(params)  # so as not to lose data if a run fails
        experiment_data.append(params)
    with open(r"runs/run_017.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

def string_to_numpy_array(array_string):
    split_string = array_string.replace("\n  ", " ").replace('.', '').replace("[", "").replace("]", "").split("\n ")
    arr = []
    for row in split_string:
        arr.append(list(map(int, row.split(" "))))
    return np.array(arr)

def get_action_vector_for_admg():
    # below string are taken from run 17 - ground truth - first element in the list
    d_str = "[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]\n [0. 1. 0. 0. 1. 0. 0. 1. 0. 0.]\n [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 1. 1. 1. 0. 0. 0. 0. 1.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]\n [0. 0. 1. 0. 1. 0. 0. 0. 0. 0.]]"
    b_str = "[[0. 1. 0. 1. 0. 0. 0. 0. 0. 0.]\n [1. 0. 0. 0. 0. 0. 0. 1. 0. 0.]\n [0. 0. 0. 1. 1. 0. 0. 0. 0. 0.]\n [1. 0. 1. 0. 0. 0. 0. 0. 1. 0.]\n [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 1. 0. 0. 0. 1. 0. 0. 0. 0.]\n [0. 0. 0. 1. 0. 0. 0. 0. 0. 1.]\n [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]"
    D = string_to_numpy_array(d_str)
    B = string_to_numpy_array(b_str)
    nx_graph = nx.from_numpy_array(D.T, create_using=nx.DiGraph)
    p = np.argsort(np.array(list(nx.topological_sort(nx_graph))))
    d = 10
    tril_ind = np.tril_indices(d, -1)
    D_sym = D + D.T
    diE = np.where(D_sym[tril_ind] > 0, 1, -1)
    B_sym = B + B.T  # should not be needed since B should be symmetric, but no harm even if it is
    biE = np.where(B_sym[tril_ind] > 0, 1, -1)
    z = np.concatenate([p, diE, biE])
    pred_D, pred_B = vec_2_bow_free_admg(z, d, tril_ind, None)
    assert np.array_equal(pred_D.astype(int), D), "pred_D and D are different"
    assert np.array_equal(pred_B, B), "pred_B and B are different"
    print('ground truth bic', 22982.04935524732)

def plot_some_pred_true_graphs():
    # i want to see what the graphs look like - how are the predicted graphs wrong - is there some pattern to them? - so I will be trying to plot some 10 graphs from somewhere - hopefully with good variety for num nodes and sample size
    # run 11 - 5 node, 15 node, 30 node, 2000 samples, bow-free
    # run 12 - 500 samples, 4000 samples, 10 node, bow-free
    # run 13 - 15 node, 2000 samples, ancestral
    # pred_D = string_to_numpy_array("[[0. 0. 0. 0. 0.]\n [1. 0. 0. 0. 0.]\n [1. 0. 0. 1. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 1. 1. 0.]]")
    # pred_B = string_to_numpy_array("[[1. 0. 0. 1. 1.]\n [0. 1. 0. 1. 0.]\n [0. 0. 1. 0. 0.]\n [1. 1. 0. 1. 0.]\n [1. 0. 0. 0. 1.]]")
    # true_D = string_to_numpy_array("[[0. 1. 0. 1. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 1. 0.]]")
    # true_B = string_to_numpy_array("[[0. 0. 1. 0. 1.]\n [0. 0. 1. 1. 1.]\n [1. 1. 0. 1. 1.]\n [0. 1. 1. 0. 0.]\n [1. 1. 1. 0. 0.]]")
    with open('runs/run_011_stdout.json', 'r') as f:
        data = json.load(f)
    dones = {5: False, 10: False, 15: False, 30: False}
    for exp in data:
        if exp['num_nodes'] in dones and not dones[exp['num_nodes']]:
            pred_D = string_to_numpy_array(exp['relcadilac_directed_thresh_adj'])
            pred_B = string_to_numpy_array(exp['relcadilac_bidirected_thresh_adj'])
            true_D = string_to_numpy_array(exp['ground_truth_directed_adj'])
            true_B = string_to_numpy_array(exp['ground_truth_bidirected_adj'])
            draw_admg(pred_D, pred_B, f'run_11_nodes_{exp["num_nodes"]}_01_pred', 'diagrams/pred_true_graphs/')
            draw_admg(true_D, true_B, f'run_11_nodes_{exp["num_nodes"]}_01_true', 'diagrams/pred_true_graphs/')
            dones[exp['num_nodes']] = True
    with open('runs/run_012.json', 'r') as f:
        data = json.load(f)
    dones = {500: False, 2000: False, 4000: False}
    for exp in data:
        if exp['num_samples'] in dones and not dones[exp['num_samples']]:
            pred_D = string_to_numpy_array(exp['relcadilac_directed_thresh_adj'])
            pred_B = string_to_numpy_array(exp['relcadilac_bidirected_thresh_adj'])
            true_D = string_to_numpy_array(exp['ground_truth_directed_adj'])
            true_B = string_to_numpy_array(exp['ground_truth_bidirected_adj'])
            draw_admg(pred_D, pred_B, f'run_12_samples_{exp["num_samples"]}_01_pred', 'diagrams/pred_true_graphs/')
            draw_admg(true_D, true_B, f'run_11_samples_{exp["num_samples"]}_01_true', 'diagrams/pred_true_graphs/')
            dones[exp['num_samples']] = True

def check_dag_recovery(seed):
    explanation = "In run 21, I am checking the impact of setting n_envs = 16 keeping rest same from run 20"
    print(f"\n\n CHECK DAG RECOVERY \n\n {explanation}\n\n")
    params = {'num_nodes': 10, 'avg_degree': 4, 'num_samples': 2000, 'beta_low': 0.5, 'beta_high': 2.0, 'standardize_data': False, 'center_data': True, 'steps_per_env': 20_000, 'n_envs': 16, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 1, 'ent_coef': 0.05, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.3, 'generator_seed': seed, 'explanation': explanation, 'admg_model': 'bow-free'}
    D, X, bic = generator.get_lin_gauss_ev_dag(num_nodes=params['num_nodes'], avg_degree=params['avg_degree'], plot=False, do_sampling=True, num_samples=params['num_samples'], sampling_params={'beta_low': params['beta_low'], 'beta_high': params['beta_high'], 'standardize_data': params['standardize_data'], 'center_data': params['center_data']})
    B, S = np.zeros((params['num_nodes'], params['num_nodes'])), np.cov(X.T)
    print(f'\n\nGround truth BIC: {bic}\n\n')
    rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 0, 'ent_coef': params['ent_coef']}
    start = time.perf_counter()
    pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=1)
    params['relcadilac_time_sec'] = time.perf_counter() - start
    if params['do_thresholding']:
        thresh_D, thresh_B = get_thresholded_admg(pred_D, pred_B, X, S, threshold=params['threshold'])
        thresh_pag = convert_admg_to_pag(thresh_D, thresh_B)
        params['relcadilac_directed_thresh_adj'] = np.array2string(thresh_D)
        params['relcadilac_bidirected_thresh_adj'] = np.array2string(thresh_B)
        params['relcadilac_thresh_admg_metrics'] = get_admg_metrics((D, B), (thresh_D, thresh_B))
        # params['relcadilac_thresh_pag_metrics'] = get_pag_metrics(pag, thresh_pag)
    params['relcadilac_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
    # params['relcadilac_pag_metrics'] = get_pag_metrics(pag, pred_pag)
    params['ground_truth_bic'] = bic
    params['relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
    params['relcadilac_pred_bic'] = pred_bic
    params['ground_truth_directed_adj'] = np.array2string(D)
    params['ground_truth_bidirected_adj'] = np.array2string(B)
    params['relcadilac_directed_pred_adj'] = np.array2string(pred_D)
    params['relcadilac_bidirected_pred_adj'] = np.array2string(pred_B)
    print(f'\trelcadilac done')
    # print(params)
    with open('runs/run_021.json', 'w') as f:
        json.dump(params, f, indent=2)

def plot_get_dag_rewards():
    # there were ALIAS-like DAG runs in run_018, run_019, run_020, run_021. I will be using run_020 since that was the best
    with open('runs/run_020.json', 'r') as f:
        data = json.load(f)
    avg_rewards = list(map(float, data['relcadilac_avg_rewards']))
    plt.plot(avg_rewards)
    plt.xlabel("8 Steps = 1 unit")
    plt.ylabel("reward value")
    plt.title("ALIAS-DAG run, 10 node, degree 4, 2000 samples,\n20_000 steps_per_env, 8 n_envs, 1 n_steps")
    plt.show()

if __name__ == '__main__':
    # seed = random.randint(1, 100)
    seed = 28
    generator = GraphGenerator(seed)
    single_test_04(seed)
