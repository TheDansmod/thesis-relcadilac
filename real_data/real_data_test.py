# data is taken from: https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz
# the true graph is taken from diagram 2 in DOI: 10.1126/science.1105809 Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data by Sachs et al.

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json

import pandas as pd
import numpy as np

from relcadilac.utils import get_bic, plot_rewards, draw_admg_named_vertices, get_thresholded_admg, convert_admg_to_pag
from relcadilac.relcadilac import relcadilac as rel_admg
from relcadilac.metrics import get_admg_metrics, get_pag_metrics

def run_relcadilac_on_sachs(data_path):
    X = pd.read_csv(data_path).to_numpy()
    X = X - np.mean(X, axis=0)  # centering
    S = np.cov(X.T)
    D = np.zeros((11, 11))
    B = np.zeros((11, 11))
    B[3, 4] = 1  
    B[4, 3] = 1
    D[1, 0] = 1
    D[5, 1] = 1
    D[3, 2] = 1
    D[8, 2] = 1
    D[6, 4] = 1
    D[0, 7] = 1
    D[5, 7] = 1
    D[6, 7] = 1
    D[9, 7] = 1
    D[10, 7] = 1
    D[0, 8] = 1
    D[3, 8] = 1
    D[9, 8] = 1
    D[10, 8] = 1
    ground_truth_bic = get_bic(D, B, X, S)
    ground_truth_pag = convert_admg_to_pag(D, B)
    print(f'{ground_truth_bic=}')
    # using bow-free model
    params = {'num_nodes': 11, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 500, 'admg_model': 'bow-free', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 2000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 16, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0}
    # relcadilac
    rl_params = {'normalize_advantage': params['normalize_advantage'], 'n_epochs': params['n_epochs'], 'device': params['device'], 'n_steps': params['n_steps'], 'verbose': 0, 'ent_coef': params['ent_coef']}
    start = time.perf_counter()
    pred_D, pred_B, pred_pag, avg_rewards, pred_bic = rel_admg(X, S, params['admg_model'], steps_per_env=params['steps_per_env'], n_envs=params['n_envs'], rl_params=rl_params, random_state=params['vec_envs_random_state'], verbose=1)
    params['relcadilac_time_sec'] = time.perf_counter() - start
    params['relcadilac_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
    params['relcadilac_pag_metrics'] = get_pag_metrics(ground_truth_pag, pred_pag)
    params['ground_truth_bic'] = ground_truth_bic
    params['relcadilac_avg_rewards'] = list(map(str, avg_rewards['average_rewards']))
    params['relcadilac_pred_bic'] = pred_bic
    print(f'{pred_bic=}')
    print(params)
    # with open(r"runs/run_008.json", "w") as f:
    #     json.dump(params, f, indent=2)
    thresh_D, thresh_B = get_thresholded_admg(pred_D, pred_B, X, S)
    draw_admg_named_vertices(pred_D, pred_D, 'Raf,Mek,Plcg,PIP2,PIP3,Erk,Akt,PKA,PKC,P38,Jnk'.split(','), file_name='sachs_pred_admg', folder='diagrams')
    draw_admg_named_vertices(thresh_D, thresh_B, 'Raf,Mek,Plcg,PIP2,PIP3,Erk,Akt,PKA,PKC,P38,Jnk'.split(','), file_name='sachs_thresholded_admg', folder='diagrams')

def plot_reward_graph():
    with open(r'runs/run_008.json', 'r') as f:
        str_rewards = json.load(f)['relcadilac_avg_rewards']
    avg_rewards = list(map(float, str_rewards))
    plot_rewards(avg_rewards, r'diagrams/sachs_dataset_rewards.png')

if __name__ == '__main__':
    data_path = r'real_data/sachs.data.csv'
    run_relcadilac_on_sachs(data_path)
