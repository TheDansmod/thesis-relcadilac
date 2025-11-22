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

# from relcadilac.data_generator import GraphGenerator
# from relcadilac.relcadilac import relcadilac as rel_admg
# from relcadilac.metrics import get_admg_metrics, get_pag_metrics
# from relcadilac.utils import draw_admg, get_ananke_bic
# from gfci.gfci import gfci_search
# from dcd.admg_discovery import Discovery

def num_nodes_variation():
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
            pred_D, pred_B, pred_pag, _ = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=0)
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
    flat_data = {}
    stack = [(data, '')]
    while stack:
        curr_dict, curr_key = stack.pop()
        for k, val in curr_dict.items():
            new_key = f"{curr_key}_{k}" if curr_key else k
            if isinstance(val, dict):
                stack.append((val, new_key))
            else:
                flat_data[new_key] = val
    return flat_data

def get_df_from_runs(data):
    rows = []
    for exp in data:
        rows.append(flatten_data(exp))
    return pd.DataFrame.from_records(rows)

def create_sample_size_plots(file_path='runs/run_004.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = get_df_from_runs(data)
    averaged_rows = []
    sample_sizes = [500, 1000, 2000, 4000]
    for sample_size in sample_sizes:
        averaged_rows.append(df.query(f'num_samples == {sample_size}').mean(axis=0, numeric_only=True))
    df_new = pd.DataFrame(averaged_rows)
    # tpr fdr f1
    fig, ax = plt.subplots(figsize=(16, 9))
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
    ax.set_title('DCD vs Relcadilac, ADMG Metrics, Ancestral Graphs')
    plt.savefig('diagrams/dcd_rel_tpr_fdr_f1_admg.png')
    plt.close()
    # skeleton tpr fdr f1
    fig, ax = plt.subplots(figsize=(16, 9))
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
    ax.set_title('DCD vs Relcadilac, ADMG Skeleton Metrics, Ancestral Graphs')
    plt.savefig('diagrams/dcd_rel_tpr_fdr_f1_admg_skeleton.png')
    plt.close()
    # shd runtime
    fig, ax = plt.subplots(figsize=(16, 9))
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
    ax.set_title('DCD vs Relcadilac, ADMG SHD and Runtime, Ancestral Graphs')
    plt.savefig('diagrams/dcd_rel_shd_runtime.png')
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
        ax.set_title('DCD vs Relcadilac vs GFCI, PAG Metrics, Ancestral Graphs')
        plt.savefig(f'diagrams/dcd_rel_gfci_{tp}_pag_metrics.png')
        plt.close()

def create_num_nodes_plot(file_path='runs/run_003.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = get_df_from_runs(data)
    # df.to_csv('runs/run_003.csv')
    averaged_rows = []
    num_nodes_list = [5, 10, 15, 20, 30]
    for num_nodes in num_nodes_list:
        averaged_rows.append(df.query(f'num_nodes == {num_nodes}').mean(axis=0, numeric_only=True))
    df_new = pd.DataFrame(averaged_rows)
    # df_new.to_csv('runs/run_003_avg.csv')
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
    ax.set_title('DCD vs Relcadilac, ADMG Metrics, Ancestral Graphs')
    plt.savefig('diagrams/dcd_rel_tpr_fdr_f1_admg.png')
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
    ax.set_title('DCD vs Relcadilac, ADMG Skeleton Metrics, Ancestral Graphs')
    plt.savefig('diagrams/dcd_rel_tpr_fdr_f1_admg_skeleton.png')
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
    ax.set_title('DCD vs Relcadilac, ADMG SHD and Runtime, Ancestral Graphs')
    plt.savefig('diagrams/dcd_rel_shd_runtime.png')
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
        ax.set_title('DCD vs Relcadilac vs GFCI, PAG Metrics, Ancestral Graphs')
        plt.savefig(f'diagrams/dcd_rel_gfci_{tp}_pag_metrics.png')
        plt.close()
        print(f'dcd_rel_gfci_{tp}_pag_metrics plotted')

if __name__ == '__main__':
    seed = 20
    # generator = GraphGenerator(seed)
    create_num_nodes_plot()
