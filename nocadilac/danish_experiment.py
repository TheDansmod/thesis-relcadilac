import random

from relcadilac.data_generator import GraphGenerator

def run_tests(seed):
    explanation = "The previous run went well, but I think the ground truth graph was really disconnected which caused it to go well since there were few edges. This time I am trying with 20 nodes and am requiring connected. Want to see how long the run will take and if it will perform just as well."
    print(f" \n\n SINGLE TEST 04 \n\n {explanation}\n\n")
    params = {'num_nodes': 20, 'avg_degree': 4, 'frac_directed': 0.6, 'degree_variance': 0.2, 'num_samples': 2000, 'admg_model': 'ancestral', 'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'standardize_data': False, 'center_data': True, 'steps_per_env': 20_000, 'n_envs': 8, 'normalize_advantage': True, 'n_epochs': 1, 'device': 'cuda', 'n_steps': 1, 'ent_coef': 0.05, 'dcd_num_restarts': 1, 'vec_envs_random_state': 0, 'do_thresholding': True, 'threshold': 0.05, 'generator_seed': seed, 'explanation': explanation, 'topo_order_known': False, 'use_logits_partition': False, 'get_pag': True, 'require_connected': False, 'run_number': 30, 'use_sde': True, 'do_entropy_annealing': False, 'initial_entropy': 0.3, 'min_entropy': 0.005, 'cycle_length': 20_000, 'damping_factor': 0.5, 'run_commit': '6c1b98079c50600dad7c3172f07049ef54e0d120'}
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

if __name__ == '__main__':
    seed = random.randint(1, 100)
    generator = GraphGenerator(seed)
    run_tests(seed)
