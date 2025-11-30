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
import dcd.admg_discovery as dcd
import relcadilac.relcadilac as relcd

class Experiments:
    def __init__(self):
        self.algorithm_name = "DCD"  # should be one of DCD or CMA-ES or Relcadilac
        self.algorithm = self.get_algorithm()
        self.run_commit = "4a68fb651ec2ab138e13a52b4a4ba97cac59fe5c"

        self.log_file = Path('runs/runs-copy.csv')
        self.log_df = pd.read_csv(self.log_file)
        self.set_run_number()
        self.data_file = Path(f'runs/run_{self.run_number:03}_data.pkl')

        self.sachs_data = True
        self.set_graph_generation_params()
        self.set_post_prediction_params()
        self.set_relcadilac_params()
        self.set_cmaes_params()
        self.set_dcd_params()
        self.set_metrics_to_none()  # so that even if an algorithm does not provide some metric, we can still put it into the table

        self.explanation = f"No explanation given."

    def set_graph_generation_params(self):
        self.generator_seed = random.randint(1, pow(10, 6))
        self.num_nodes = 10
        self.avg_degree = 4
        self.frac_directed = 0.6
        self.degree_variance = 0.0
        self.num_samples = 2000
        self.admg_model = 'bow-free'  # could be one of 'ancestral' or 'bow-free'
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
        if self.algorithm_name == 'CMA_ES':
            path.mkdir(exist_ok=True)
        self.cmaes_output_folder = f"{path}{os.sep}"
        self.cmaes_lambda = 1e-4
        self.cmaes_delta = 1.0
        self.cmaes_gamma = 0.2 * np.log(self.num_samples) / (self.num_nodes * self.cmaes_delta)
        self.cmaes_obj_fn_type = 'order_edge_stability'  # can be one of order_edge_stability or z_l2_regularization

    def set_relcadilac_params(self):
        self.steps_per_env = 10_000
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
        self.do_entropy_annealing = True
        self.initial_entropy = 0.3
        self.min_entropy = 0.005
        self.cycle_length = 16_000
        self.damping_factor = 0.5

    def set_dcd_params(self):
        self.dcd_num_restarts = 5

    def get_algorithm(self):
        if self.algorithm_name == 'CMA-ES':
            return cma_es.cmaes_admg_search
        elif self.algorithm_name == 'DCD':
            dcd_admg_search = dcd.Discovery().discover_admg
            return dcd_admg_search
        elif self.algorithm_name == 'Relcadilac':
            return relcd.relcadilac

    def set_run_number(self):
        if len(self.log_df) == 0:
            self.run_number = 30
        else:
            self.run_number = self.log_df['run_number'].max() + 1

    def set_metrics_to_none(self):
        # this could probably be done better
        self.thresh_admg_tpr = None
        self.thresh_admg_fdr = None
        self.thresh_admg_f1 = None
        self.thresh_admg_shd = None
        self.thresh_admg_skeleton_tpr = None
        self.thresh_admg_skeleton_fdr = None
        self.thresh_admg_skeleton_f1 = None
        self.thresh_pag_skeleton_f1 = None
        self.thresh_pag_skeleton_tpr = None
        self.thresh_pag_skeleton_fdr = None
        self.thresh_pag_circle_f1 = None
        self.thresh_pag_circle_tpr = None
        self.thresh_pag_circle_fdr = None
        self.thresh_pag_head_f1 = None
        self.thresh_pag_head_tpr = None
        self.thresh_pag_head_fdr = None
        self.thresh_pag_tail_f1 = None
        self.thresh_pag_tail_tpr = None
        self.thresh_pag_tail_fdr = None
        self.admg_tpr = None
        self.admg_fdr = None
        self.admg_f1 = None
        self.admg_shd = None
        self.admg_skeleton_tpr = None
        self.admg_skeleton_fdr = None
        self.admg_skeleton_f1 = None
        self.pag_skeleton_f1 = None
        self.pag_skeleton_tpr = None
        self.pag_skeleton_fdr = None
        self.pag_circle_f1 = None
        self.pag_circle_tpr = None
        self.pag_circle_fdr = None
        self.pag_head_f1 = None
        self.pag_head_tpr = None
        self.pag_head_fdr = None
        self.pag_tail_f1 = None
        self.pag_tail_tpr = None
        self.pag_tail_fdr = None
        self.thresh_pred_bic = None
        self.pred_bic = None
        self.true_bic = None
        self.runtime = None

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

    def set_sachs_data(self):
        path = 'real_data/sachs.data.csv'
        X = pd.read_csv(path).to_numpy()
        self.num_samples, self.num_nodes = X.shape
        self.data = X - np.mean(X, axis=0)  # centering
        self.data_cov = np.cov(X.T)
        D = np.zeros((self.num_nodes, self.num_nodes))
        B = np.zeros((self.num_nodes, self.num_nodes))
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
        D[2, 4] = 1
        D[9, 7] = 1
        D[10, 7] = 1
        D[0, 8] = 1
        D[8, 3] = 1
        D[9, 8] = 1
        D[10, 8] = 1
        self.avg_degree = 15 * 2 / self.num_nodes
        self.frac_directed = 14 / 15
        self.true_D, self.true_B = D, B
        self.true_bic = utils.get_bic(self.true_D, self.true_B, self.data, self.data_cov)
        self.true_pag = utils.convert_admg_to_pag(self.true_D, self.true_B)

    def get_algorithm_params(self):
        if self.algorithm_name == 'CMA-ES':
            return {'max_fevals': self.max_fevals, 'verbose': self.cmaes_verbose_level, 'popsize': self.cmaes_popsize, 'num_parallel_workers': self.cmaes_num_parallel_workers, 'output_folder': self.cmaes_output_folder, 'cmaes_lambda': self.cmaes_lambda, 'gamma': self.cmaes_gamma, 'delta': self.cmaes_delta, 'obj_fn_type': self.cmaes_obj_fn_type}
        elif self.algorithm_name == 'DCD':
            return {'num_restarts': self.dcd_num_restarts, 'local': False, 'verbose': True}
        elif self.algorithm_name == 'Relcadilac':
            rl_params = {"normalize_advantage": self.normalize_advantage, "n_epochs": self.n_epochs, "device": self.device, "verbose": 0, 'n_steps': self.n_steps, 'ent_coef': self.ent_coef, 'use_sde': self.use_sde}
            entropy_params = {'initial_entropy': self.initial_entropy, 'min_entropy': self.min_entropy, 'cycle_length': self.cycle_length, 'damping_factor': self.damping_factor}
            # the None for topo order should be fixed, currently it is difficult to pass a topo order
            return {'steps_per_env': self.steps_per_env, 'n_envs': self.n_envs, 'rl_params': rl_params, 'verbose': 1, 'random_state': self.vec_envs_random_state, 'topo_order': None, 'use_logits_partition': self.use_logits_partition, 'do_entropy_annealing': self.do_entropy_annealing, 'entropy_params': entropy_params}

    def plot_admgs(self):
        file_prefix = f'run_{self.run_number:03}_{self.admg_model}_'
        directory = Path('diagrams')
        if not self.sachs_data:
            utils.draw_admg(self.true_D, self.true_B, f'{file_prefix}true', directory)
            utils.draw_admg(self.pred_D, self.pred_B, f'{file_prefix}pred', directory)
            utils.draw_admg(self.pred_thresh_D, self.pred_thresh_B, f'{file_prefix}pred_thresh', directory)
        else:
            vertex_names = 'Raf,Mek,Plcg,PIP2,PIP3,Erk,Akt,PKA,PKC,P38,Jnk'.split(',')
            utils.draw_admg_named_vertices(self.true_D, self.true_B, vertex_names, f'{file_prefix}true', directory)
            utils.draw_admg_named_vertices(self.pred_D, self.pred_B, vertex_names, f'{file_prefix}pred', directory)
            utils.draw_admg_named_vertices(self.pred_thresh_D, self.pred_thresh_B, vertex_names, f'{file_prefix}pred_thresh', directory)

    def log_metrics_and_data(self):
        # data
        data_dict = {'true_D': self.true_D, 'true_B': self.true_B, 'data': self.data, 'data_cov': self.data_cov, 'pred_D': self.pred_D, 'pred_B': self.pred_B, 'pred_thresh_D': self.pred_thresh_D if self.do_thresholding else None, 'pred_thresh_B': self.pred_thresh_B if self.do_thresholding else None, 'captured_metrics': self.captured_metrics}
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        # first
        f = [self.run_number, self.algorithm_name]
        # params
        p = [self.run_commit, self.explanation, self.data_file, self.log_file, self.generator_seed, self.num_nodes, self.avg_degree, self.frac_directed, self.degree_variance, self.num_samples, self.admg_model, self.beta_low, self.beta_high, self.omega_offdiag_low, self.omega_offdiag_high, self.omega_diag_low, self.omega_diag_high, self.standardize_data, self.center_data, self.get_pag, self.require_connected, self.do_thresholding, self.threshold, self.max_fevals, self.cmaes_verbose_level, self.popsize_ratio, self.cmaes_popsize, self.cmaes_num_parallel_workers, self.cmaes_output_folder, self.cmaes_lambda, self.cmaes_gamma, self.cmaes_delta, self.cmaes_obj_fn_type, self.steps_per_env, self.n_envs, self.normalize_advantage, self.n_epochs, self.device, self.n_steps, self.ent_coef, self.vec_envs_random_state, self.topo_order_known, self.use_logits_partition, self.use_sde, self.do_entropy_annealing, self.initial_entropy, self.min_entropy, self.cycle_length, self.damping_factor, self.dcd_num_restarts]
        # metrics
        m = [self.thresh_admg_tpr, self.thresh_admg_fdr, self.thresh_admg_f1, self.thresh_admg_shd, self.thresh_admg_skeleton_tpr, self.thresh_admg_skeleton_fdr, self.thresh_admg_skeleton_f1, self.thresh_pag_skeleton_f1, self.thresh_pag_skeleton_tpr, self.thresh_pag_skeleton_fdr, self.thresh_pag_circle_f1, self.thresh_pag_circle_tpr, self.thresh_pag_circle_fdr, self.thresh_pag_head_f1, self.thresh_pag_head_tpr, self.thresh_pag_head_fdr, self.thresh_pag_tail_f1, self.thresh_pag_tail_tpr, self.thresh_pag_tail_fdr, self.admg_tpr, self.admg_fdr, self.admg_f1, self.admg_shd, self.admg_skeleton_tpr, self.admg_skeleton_fdr, self.admg_skeleton_f1, self.pag_skeleton_f1, self.pag_skeleton_tpr, self.pag_skeleton_fdr, self.pag_circle_f1, self.pag_circle_tpr, self.pag_circle_fdr, self.pag_head_f1, self.pag_head_tpr, self.pag_head_fdr, self.pag_tail_f1, self.pag_tail_tpr, self.pag_tail_fdr, self.thresh_pred_bic, self.pred_bic, self.true_bic, self.runtime]
        # extra
        ex = [self.pred_bic_excess, self.thresh_pred_bic_excess, self.sachs_data]
        m = [round(val, 4) for val in m]
        self.log_df.loc[len(self.log_df)] = f + m + p + ex
        self.log_df.to_csv(self.log_file, index=False)

    def evaluate_and_set_metrics(self):
        if self.algorithm_name in ['CMA-ES', 'Relcadilac', 'DCD']:
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
                self.thresh_pred_bic_excess = (self.thresh_pred_bic - self.true_bic) / self.true_bic
            m = metrics.get_admg_metrics((self.true_D, self.true_B), (self.pred_D, self.pred_B))
            self.admg_tpr, self.admg_fdr, self.admg_f1, self.admg_shd = m['admg']['tpr'], m['admg']['fdr'], m['admg']['f1'], m['admg']['shd']
            self.admg_skeleton_tpr, self.admg_skeleton_fdr, self.admg_skeleton_f1 = m['skeleton']['tpr'], m['skeleton']['fdr'], m['skeleton']['f1']
            m = metrics.get_pag_metrics(self.true_pag, self.pred_pag)
            self.pag_skeleton_f1, self.pag_skeleton_tpr, self.pag_skeleton_fdr = m['skeleton']['f1'], m['skeleton']['tpr'], m['skeleton']['fdr']
            self.pag_circle_f1, self.pag_circle_tpr, self.pag_circle_fdr = m['circle']['f1'], m['circle']['tpr'], m['circle']['fdr']
            self.pag_head_f1, self.pag_head_tpr, self.pag_head_fdr = m['head']['f1'], m['head']['tpr'], m['head']['fdr']
            self.pag_tail_f1, self.pag_tail_tpr, self.pag_tail_fdr = m['tail']['f1'], m['tail']['tpr'], m['tail']['fdr']
            self.pred_bic_excess = (self.pred_bic - self.true_bic) / self.true_bic

    def run_test(self):
        try:
            print(f'\nRUN NUMBER: {self.run_number}\nEXPLANATION: {self.explanation}\n')
            if self.sachs_data:
                self.set_sachs_data()
            else:
                self.set_data()
            print(f'\nTRUE BIC: {self.true_bic}\n')
            start = time.perf_counter()
            self.pred_D, self.pred_B, self.pred_pag, self.pred_bic, self.captured_metrics = self.algorithm(self.data, self.data_cov, self.admg_model, **self.get_algorithm_params())
            self.runtime = time.perf_counter() - start
            self.evaluate_and_set_metrics()
            print(f'\nPredicted D:\n{self.pred_D}\nPredicted B:\n{self.pred_B}\nTrue bic: {self.true_bic}\nPredicted bic: {self.pred_bic}\nThresholded bic: {self.thresh_pred_bic}\nThresholded SHD: {self.thresh_admg_shd}\nPredicted ADMG SHD: {self.admg_shd}\n'), 
            self.log_metrics_and_data()
            self.plot_admgs()
        except Exception as e:
            print("THERE WAS AN EXCEPTION")
            traceback.print_exc()
            print("CONTINUING ANYWAY")

def run_variation_test_01():
    # with this test, for each of ancestral and bow-free admg types, I am varying the number of nodes while keeping the sample size at 2k, and then varying the sample size while keeping the number of nodes at 10
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
                exp.run_test()
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
                exp.run_test()

def run_cmaes_obj_fn_test():
    # I have previously not used the stability modification to the objective function, so I am running it a few times so that I can compare it with the z_l2_regularization approach which I had been using
    repetitions = 3
    admg_models = ['ancestral', 'bow-free']
    for repeat in range(repetitions):
        for curr_model in admg_models:
            exp = Experiments()
            exp.admg_model = curr_model
            exp.explanation = f"Run {exp.run_number}; {exp.algorithm_name}; Checking out the stability modification to the objective function. I have updated the value of gamma to be more theoretically consistent and made it larger so it actually has an impact but not made it so large that it becomes greater than the benefit of a single edge. The value of gamma is {exp.cmaes_gamma}. Test suite varies the admg model. I have increased the proportion of directed edges. I think this might help. This is iteration {repeat + 1} of {repetitions} with admg model as {exp.admg_model}. Using {exp.num_nodes} nodes and {exp.num_samples} samples. The maximum number of evaluations is {exp.max_fevals}."
            exp.run_test()

def run_variation_test_02():
    algos = ['CMA-ES', 'DCD', 'Relcadilac']
    frac_directed_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    avg_degree_list = [2, 3, 4, 6, 8]
    admg_models = ['ancestral', 'bow-free']
    it = 1
    for curr_model in admg_models:
        for curr_algo in algos:
            for curr_frac in frac_directed_list:
                exp = Experiments()
                exp.frac_directed = curr_frac
                exp.algorithm_name = curr_algo
                exp.admg_model = curr_model
                exp.algorithm = exp.get_algorithm()
                exp.explanation = f"Run {exp.run_number}; {exp.algorithm_name}; Frac directed = {exp.frac_directed}; Varying directed fraction. {it} of 60. First run in test = 210."
                exp.run_test()
                it += 1
            for curr_deg in avg_degree_list:
                exp = Experiments()
                exp.avg_degree = curr_deg
                exp.algorithm_name = curr_algo
                exp.admg_model = curr_model
                exp.algorithm = exp.get_algorithm()
                exp.explanation = f"Run {exp.run_number}; {exp.algorithm_name}; Avg degree = {exp.avg_degree}; Varying average degree. {it} of 60. First run in test = 210."
                exp.run_test()
                it += 1

def run_sachs_dataset():
    for algo in ['DCD', 'CMA-ES', 'Relcadilac']:
        exp = Experiments()
        exp.algorithm_name = algo
        exp.algorithm = exp.get_algorithm()
        exp.explanation = f"Run {exp.run_number}; {exp.algorithm_name}; Running DCD, CMA-ES, Relcadilac for sachs dataset. First run = 271"
        exp.run_test()
    
if __name__ == '__main__':
    run_sachs_dataset()
