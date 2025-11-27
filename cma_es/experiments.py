import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import random
import pickle
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

import cma_es.cma_es as cma_es
import relcadilac.utils as utils
import relcadilac.metrics as metrics
import relcadilac.data_generator as data_generator

class Experiments:
    def __init__(self):
        self.algorithm_name = "CMA-ES"
        self.algorithm = self.get_algorithm()
        self.run_commit = "4152a0d1b5e38cc4612f3a6bba5641e8b9f545df"

        self.log_file = Path('runs/runs.csv')
        self.log_df = pd.read_csv(self.log_file)
        self.set_run_number()
        self.data_file = Path(f'runs/run_{self.run_number:03}_data.pkl')

        self.set_graph_generation_params()
        self.set_post_prediction_params()
        self.set_relcadilac_params()
        self.set_cmaes_params()
        self.set_dcd_params()

        self.explanation = f"No explanation given."

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
        self.max_fevals = 40_000
        self.cmaes_verbose_level = 3
        self.popsize_ratio = 4
        self.cmaes_popsize = int((4 + 3 * np.log(self.num_nodes * self.num_nodes)) * self.popsize_ratio)
        self.cmaes_num_parallel_workers = 12
        path = Path(f'runs/cmaes_{self.run_number:03}/')
        path.mkdir(exist_ok=True)
        self.cmaes_output_folder = f"{path}{os.sep}"

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

    def set_run_number(self):
        if len(self.log_df) == 0:
            self.run_number = 30
        else:
            self.run_number = self.log_df['run_number'].max() + 1

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
            return {'max_fevals': self.max_fevals, 'verbose': self.cmaes_verbose_level, 'popsize': self.cmaes_popsize, 'num_parallel_workers': self.cmaes_num_parallel_workers, 'output_folder': self.cmaes_output_folder}

    def plot_graphs(self):
        file_prefix = f'run_{self.run_number:03}_{self.admg_model}_'
        directory = Path('diagrams')
        utils.draw_admg(self.true_D, self.true_B, f'{file_prefix}true', directory)
        utils.draw_admg(self.pred_D, self.pred_B, f'{file_prefix}pred', directory)
        utils.draw_admg(self.pred_thresh_D, self.pred_thresh_B, f'{file_prefix}pred_thresh', directory)

    def log_metrics_and_data(self):
        # data
        data_dict = {'true_D': self.true_D, 'true_B': self.true_B, 'data': self.data, 'data_cov': self.data_cov, 'pred_D': self.pred_D, 'pred_B': self.pred_B, 'pred_thresh_D': self.pred_thresh_D if self.do_thresholding else None, 'pred_thresh_B': self.pred_thresh_B if self.do_thresholding else None, 'captured_metrics': self.captured_metrics}
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        # first
        f = [self.run_number, self.algorithm_name]
        # params
        p = [self.run_commit, self.explanation, self.data_file, self.log_file, self.generator_seed, self.num_nodes, self.avg_degree, self.frac_directed, self.degree_variance, self.num_samples, self.admg_model, self.beta_low, self.beta_high, self.omega_offdiag_low, self.omega_offdiag_high, self.omega_diag_low, self.omega_diag_high, self.standardize_data, self.center_data, self.get_pag, self.require_connected, self.do_thresholding, self.threshold, self.max_fevals, self.cmaes_verbose_level, self.popsize_ratio, self.cmaes_popsize, self.cmaes_num_parallel_workers, self.cmaes_output_folder, self.steps_per_env, self.n_envs, self.normalize_advantage, self.n_epochs, self.device, self.n_steps, self.ent_coef, self.vec_envs_random_state, self.topo_order_known, self.use_logits_partition, self.use_sde, self.do_entropy_annealing, self.initial_entropy, self.min_entropy, self.cycle_length, self.damping_factor, self.dcd_num_restarts]
        # metrics
        m = [self.thresh_admg_tpr, self.thresh_admg_fdr, self.thresh_admg_f1, self.thresh_admg_shd, self.thresh_admg_skeleton_tpr, self.thresh_admg_skeleton_fdr, self.thresh_admg_skeleton_f1, self.thresh_pag_skeleton_f1, self.thresh_pag_skeleton_tpr, self.thresh_pag_skeleton_fdr, self.thresh_pag_circle_f1, self.thresh_pag_circle_tpr, self.thresh_pag_circle_fdr, self.thresh_pag_head_f1, self.thresh_pag_head_tpr, self.thresh_pag_head_fdr, self.thresh_pag_tail_f1, self.thresh_pag_tail_tpr, self.thresh_pag_tail_fdr, self.admg_tpr, self.admg_fdr, self.admg_f1, self.admg_shd, self.admg_skeleton_tpr, self.admg_skeleton_fdr, self.admg_skeleton_f1, self.pag_skeleton_f1, self.pag_skeleton_tpr, self.pag_skeleton_fdr, self.pag_circle_f1, self.pag_circle_tpr, self.pag_circle_fdr, self.pag_head_f1, self.pag_head_tpr, self.pag_head_fdr, self.pag_tail_f1, self.pag_tail_tpr, self.pag_tail_fdr, self.thresh_pred_bic, self.pred_bic, self.runtime]
        m = [round(val, 4) for val in m]
        self.log_df.loc[len(self.log_df)] = f + m + p
        self.log_df.to_csv(self.log_file, index=False)

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
        print(f'\nTRUE BIC: {self.true_bic}\n')
        start = time.perf_counter()
        self.pred_D, self.pred_B, self.pred_pag, self.pred_bic, self.captured_metrics = self.algorithm(self.data, self.data_cov, self.admg_model, **self.get_algorithm_params())
        self.runtime = time.perf_counter() - start
        self.evaluate_and_set_metrics()
        print(f'\nPredicted D:\n{self.pred_D}\nPredicted B:\n{self.pred_B}\nPredicted bic: {self.pred_bic}\nThresholded bic: {self.thresh_pred_bic}\nThresholded SHD: {self.thresh_admg_shd}\nPredicted ADMG SHD: {self.admg_shd}\n'), 
        self.log_metrics_and_data()
        self.plot_graphs()

def run_variation_test():
    num_nodes = [5, 10, 15, 20, 30]
    sample_sizes = [500, 1000, 3000, 4000]  # since 2k is already covered
    admg_models = ['ancestral', 'bow-free']
    repetitions = 3
    for repeat in range(repetitions):
        for curr_model in admg_models:
            for curr_nodes in num_nodes:
                exp = Experiments()
                exp.num_nodes = curr_nodes
                exp.admg_model = curr_model
                exp.cmaes_popsize = int((4 + 3 * np.log(exp.num_nodes * exp.num_nodes)) * exp.popsize_ratio)
                if curr_nodes == 5:
                    exp.max_fevals = 20_000
                else:
                    exp.max_fevals = 40_000
                exp.explanation = f"Run {exp.run_number}; {exp.algorithm_name}; Number of nodes: {exp.num_nodes}; Sample size: {exp.num_samples}; Fn Evals: {exp.max_fevals}; ADMG Model: {exp.admg_model}; Test suite that varies number of nodes and number of samples. This is iteration {repeat + 1} of {repetitions} varying number of nodes for {exp.admg_model} graphs."
                try:
                    exp.run_test()
                except Exception as e:
                    print("THERE WAS AN EXCEPTION")
                    traceback.print_exc()
                    print("CONTINUING ANYWAY")
            for curr_samples in sample_sizes:
                exp = Experiments()
                exp.num_samples = curr_samples
                exp.admg_model = curr_model
                exp.cmaes_popsize = int((4 + 3 * np.log(exp.num_nodes * exp.num_nodes)) * exp.popsize_ratio)
                if curr_samples == 500:
                    exp.max_fevals = 20_000
                else:
                    exp.max_fevals = 40_000
                exp.explanation = f"Run {exp.run_number}; {exp.algorithm_name}; Number of nodes: {exp.num_nodes}; Sample size: {exp.num_samples}; Fn Evals: {exp.max_fevals}; ADMG Model: {exp.admg_model}; Test suite that varies number of nodes and number of samples. This is iteration {repeat + 1} of {repetitions} varying number of samples for {exp.admg_model} graphs."
                try:
                    exp.run_test()
                except Exception as e:
                    print("THERE WAS AN EXCEPTION")
                    traceback.print_exc()
                    print("CONTINUING ANYWAY")

if __name__ == '__main__':
    run_variation_test()
