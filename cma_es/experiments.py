import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
from pathlib import Path

import random
import pandas as pd
import numpy as np

import cma_es.cma_es as cma_es
import relcadilac.utils as utils
import relcadilac.metrics as metrics

class Experiments:
    def __init__(self):
        self.run_number = 30
        self.algorithm_name = "CMA-ES"
        self.algorithm = self.get_algorithm()
        self.run_commit = "060fdcce2cedd983af8f119979cdbbb22b65575e"
        self.explanation = f"Run {self.run_number}; {self.algorithm_name}; Just checking if this experiment framework functions or not. Only 5_000 function evaluations."

        self.set_graph_generation_params()
        self.set_post_prediction_params()
        self.set_relcadilac_params()
        self.set_cmaes_params()
        self.set_dcd_params()

        self.data_file = Path(f'runs/run_{self.run_number:03}_data.pkl')
        self.log_file = Path('runs/runs.csv')

    def set_graph_generation_params(self):
        self.generator_seed = random.randint(1, 100)
        self.num_nodes = 10
        self.avg_degree = 4
        self.frac_directed = 0.6
        self.degree_variance = 0.2
        self.num_samples = 2000
        self.admg_model = 'ancestral'
        self.beta_low = 0.5
        self.beta_high = 2.0
        self.omega_offdiag_low = 0.4
        self.omega_offdiag_high = 0.7
        self.omega_diag_low = 0.7
        self.omega_diag_high = 1.2
        self.standardize_data = False
        self.center_data = True
        self.get_pag = True
        self.require_connected = False
        self.generator = data_generator.GraphGenerator(self.generator_seed)

    def set_post_prediction_params(self):
        self.do_thresholding = True
        self.threshold = 0.05

    def set_cmaes_params(self):
        self.max_fevals = 5_000
        self.cmaes_verbose_level = 3
        self.popsize_ratio = 4
        self.cmaes_popsize = int((4 + 3 * np.log(self.num_nodes * self.num_nodes)) * self.popsize_ratio)
        self.cmaes_num_parallel_workers = 8
        self.output_folder = Path(f'runs/cmaes_{self.run_number:03}')
        self.output_folder.mkdir(exist_ok=True)

    def set_relcadilac_params(self):
        self.steps_per_env = 20_000
        self.n_envs = 8
        self.normalize_advantage = True
        self.n_epochs = 1
        self.device = 'cuda'
        self.n_steps = 1
        self.ent_coef = 0.05
        self.vec_envs_random_state = 1
        self.topo_order_known = False
        self.use_logits_partition = False
        self.use_sde = True
        self.do_entropy_annealing = False
        self.initial_entropy = 0.3
        self.min_entropy = 0.005
        self.cycle_length = 10_000
        self.damping_factor = 0.5

    def set_dcd_params(self):
        self.dcd_num_restarts = 1

    def get_algorithm(self):
        if self.algorithm_name == 'CMA-ES':
            return cma_es.cmaes_admg_search

    def get_sampling_params(self):
        return {'beta_low': self.beta_low, 'beta_high': self.beta_high, 'omega_offdiag_low': self.omega_offdiag_low, 'omega_offdiag_high': self.omega_offdiag_high, 'omega_diag_low': self.omega_diag_low, 'omega_diag_high': self.omega_diag_high, 'standardize_data': self.standardize_data, 'center_data': self.center_data}

    def set_data(self):
        self.true_D, self.true_B, self.data, self.data_cov, self.true_bic, self.true_pag = self.generator.get_admg(
            num_nodes=self.num_nodes,
            avg_degree=self.avg_degree,
            frac_directed=self.frac_directed,
            degree_variance=self.degree_variance,
            admg_model=self.admg_model,
            plot=False,
            do_sampling=True,
            num_samples=self.num_samples,
            sampling_params=self.get_sampling_params(),
            get_pag=self.get_pag,
            require_connected=self.require_connected,
        )

    def get_algorithm_params(self):
        if self.algorithm_name == 'CMA-ES':
            return {'max_fevals': self.max_fevals, 'verbose': self.cmaes_verbose_level, 'popsize': self.cmaes_popsize, 'num_parallel_workers': self.cmaes_num_parallel_workers, 'output_folder': self.output_folder}

    def log_metrics_and_data(self):
        # data
        data_dict = {'true_D': self.true_D, 'true_B': self.true_B, 'data': self.data, 'data_cov': self.data_cov, 'pred_D': self.pred_D, 'pred_B': self.pred_B, 'pred_thresh_D': self.pred_thresh_D if self.do_thresholding else None, 'pred_thresh_B': self.pred_thresh_B if self.do_thresholding else None}
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        # first
        f = [self.run_number, self.algorithm_name]
        # params
        p = [self.run_commit, self.explanation, self.data_file, self.log_file, self.generator_seed, self.num_nodes, self.avg_degree, self.frac_directed, self.degree_variance, self.num_samples, self.admg_model, self.beta_low, self.beta_high, self.omega_offdiag_low, self.omega_offdiag_high, self.omega_diag_low, self.omega_diag_high, self.standardize_data, self.center_data, self.get_pag, self.require_connected, self.do_thresholding, self.threshold, self.max_fevals, self.cmaes_verbose_level, self.popsize_ratio, self.cmaes_popsize, self.cmaes_num_parallel_workers, self.steps_per_env, self.n_envs, self.normalize_advantage, self.n_epochs, self.device, self.n_steps, self.ent_coef, self.vec_envs_random_state, self.topo_order_known, self.use_logits_partition, self.use_sde, self.do_entropy_annealing, self.initial_entropy, self.min_entropy, self.cycle_length, self.damping_factor, self.dcd_num_restarts]
        # metrics
        m = [self.thresh_admg_tpr, self.thresh_admg_fdr, self.thresh_admg_f1, self.thresh_admg_shd, self.thresh_admg_skeleton_tpr, self.thresh_admg_skeleton_fdr, self.thresh_admg_skeleton_f1, self.thresh_pag_skeleton_f1, self.thresh_pag_skeleton_tpr, self.thresh_pag_skeleton_fdr, self.thresh_pag_circle_f1, self.thresh_pag_circle_tpr, self.thresh_pag_circle_fdr, self.thresh_pag_head_f1, self.thresh_pag_head_tpr, self.thresh_pag_head_fdr, self.thresh_pag_tail_f1, self.thresh_pag_tail_tpr, self.thresh_pag_tail_fdr, self.admg_tpr, self.admg_fdr, self.admg_f1, self.admg_shd, self.admg_skeleton_tpr, self.admg_skeleton_fdr, self.admg_skeleton_f1, self.pag_skeleton_f1, self.pag_skeleton_tpr, self.pag_skeleton_fdr, self.pag_circle_f1, self.pag_circle_tpr, self.pag_circle_fdr, self.pag_head_f1, self.pag_head_tpr, self.pag_head_fdr, self.pag_tail_f1, self.pag_tail_tpr, self.pag_tail_fdr, self.thresh_pred_bic, self.pred_bic, self.runtime]
        df = pd.read_csv(self.log_file)
        df.loc[len(df)] = f + m + p
        df.to_csv(self.log_file)

    def evaluate_and_set_metrics(self):
        if self.do_thresholding:
            self.pred_thresh_D, self.pred_thresh_B, self.thresh_pred_bic = utils.get_thresholded_admg(self.pred_D, self.pred_B, self.data, self.data_cov, threshold=self.threshold, get_bic=True)
            self.pred_thresh_pag = utils.convert_admg_to_pag(self.pred_thresh_D, self.pred_thresh_B)
            m = metrics.get_admg_metrics((self.true_D, self.true_B), (self.pred_thresh_D, self.pred_thresh_B))
            self.thresh_admg_tpr, self.thresh_admg_fdr, self.thresh_admg_f1, self.thresh_admg_shd = m['admg']['tpr'], m['admg']['fdr'], m['admg']['f1'], m['admg']['shd']
            self.thresh_admg_skeleton_tpr, self.thresh_admg_skeleton_fdr, self.thresh_admg_skeleton_f1 = m['skeleton']['tpr'], m['skeleton']['fdr'], m['skeleton']['f1']
            m = metrics.get_pag_metrics(self.true_pag, self.pred_thresh_pag)
            self.thresh_pag_skeleton_f1, self.thresh_pag_skeleton_tpr, self.thresh_pag_skeleton_fdr = m['skeleton']['f1'], m['skeleton']['tpr'], m['skeleton']['fdr']
            self.thresh_pag_circle_f1, self.thresh_pag_circle_tpr, self.thresh_pag_circle_fdr = m['circle']['f1'], m['circle']['tpr'], m['circle']['fdr']
            self.thresh_pag_head_f1, self.thresh_pag_head_tpr, self.thresh_pag_head_fdr = m['head']['f1'], m['head']['tpr'], m['head']['fdr']
            self.thresh_pag_tail_f1, self.thresh_pag_tail_tpr, self.thresh_pag_tail_fdr = m['tail']['f1'], m['tail']['tpr'], m['tail']['fdr']
        m = metrics.get_admg_metrics((self.true_D, self.true_B), (self.pred_D, self.pred_B))
        self.admg_tpr, self.admg_fdr, self.admg_f1, self.admg_shd = m['admg']['tpr'], m['admg']['fdr'], m['admg']['f1'], m['admg']['shd']
        self.admg_skeleton_tpr, self.admg_skeleton_fdr, self.admg_skeleton_f1 = m['skeleton']['tpr'], m['skeleton']['fdr'], m['skeleton']['f1']
        m = metrics.get_pag_metrics(self.true_pag, self.pred_pag)
        self.pag_skeleton_f1, self.pag_skeleton_tpr, self.pag_skeleton_fdr = m['skeleton']['f1'], m['skeleton']['tpr'], m['skeleton']['fdr']
        self.pag_circle_f1, self.pag_circle_tpr, self.pag_circle_fdr = m['circle']['f1'], m['circle']['tpr'], m['circle']['fdr']
        self.pag_head_f1, self.pag_head_tpr, self.pag_head_fdr = m['head']['f1'], m['head']['tpr'], m['head']['fdr']
        self.pag_tail_f1, self.pag_tail_tpr, self.pag_tail_fdr = m['tail']['f1'], m['tail']['tpr'], m['tail']['fdr']

    def run_test(self):
        print(f'\nRUN NUMBER: {self.run_number}\nEXPLANATION: {self.explanation}\n')
        self.set_data()
        start = time.perf_counter()
        self.pred_D, self.pred_B, self.pred_pag, self.pred_bic = self.algorithm(data, data_cov, **self.get_algorithm_params())
        self.runtime = time.perf_counter() - start
        self.evaluate_and_set_metrics()
        self.log_metrics_and_data()

if __name__ == '__main__':
    exp = Experiments()
    exp.run_test()
