import os
import time
import logging
import warnings
from copy import deepcopy

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from relcadilac.admg_env import ADMGEnv
from relcadilac.tracking_callback import TrackingCallback
from relcadilac.utils import vec_2_bow_free_admg, vec_2_ancestral_admg, convert_admg_to_pag, vec_2_bow_free_admg_known_topo_order, vec_2_ancestral_admg_known_topo_order, vec_2_bow_free_admg_logits
from relcadilac.data_generator import GraphGenerator
# from relcadilac.utils import vec2dag  # uncomment this line for ALIAS  (1 of 7 changes to get ALIAS behaviour - search the word ALIAS in comments) - need only make changes in the relcadilac.py file and admg_env.py file, but will only get the behaviour for linear gaussian data with equal variances not non-linear like the main thrust of the ALIAS paper. Check the function check_dag_recovery in experiments to see what arguments might one use and how data generation might happen for it.

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def relcadilac(
        X,
        sample_cov,
        admg_model,   # could be either bow-free or ancestral
        steps_per_env=2000,
        n_envs=8,
        rl_params={"normalize_advantage": True, "n_epochs": 1, "device": "cuda", "verbose": 0, 'n_steps': 16, 'ent_coef': 0.05},
        verbose=1,
        random_state=0,
        topo_order=None,  # should be a numpy array of shape (d,)
        use_logits_partition=False,  # if this is true then we use the vec_2_bow_free_admg_logits function instead of the vec_2_bow_free_admg
        do_entropy_annealing=False,
        entropy_params={"initial_entropy": 0.3, "min_entropy": 0.005, "cycle_length": 10_000, "damping_factor": 0.5},
        **unused
    ):
    if not admg_model in ['bow-free', 'ancestral']:
        raise ValueError("admg_model must either be `bow-free` or `ancestral`")
    rl_params = deepcopy(rl_params or {})
    if admg_model == 'bow-free':
        if topo_order is not None:
            vec2admg = vec_2_bow_free_admg_known_topo_order 
        elif use_logits_partition:
            vec2admg = vec_2_bow_free_admg_logits
        else:
            vec2admg = vec_2_bow_free_admg
    if admg_model == 'ancestral':
        vec2admg = vec_2_ancestral_admg_known_topo_order if topo_order is not None else vec_2_ancestral_admg
    n, d = X.shape
    steps = steps_per_env * n_envs

    raw_vec_env = make_vec_env(ADMGEnv, n_envs=n_envs, env_kwargs=dict(nodes=d, X=X, sample_cov=sample_cov, vec2admg=vec2admg, topo_order=topo_order), vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(raw_vec_env, norm_obs=False, norm_reward=False, gamma=1.0, clip_reward=np.inf)
    try:
        tracking = TrackingCallback(total_timesteps=steps, num_samples=n, verbose=verbose) # , do_entropy_annealing=do_entropy_annealing, **entropy_params) danish add later

        model = PPO("MlpPolicy", vec_env, seed=random_state, **rl_params)
        if verbose:
            logger.info("triggering learn")
        model.learn(total_timesteps=steps, callback=tracking) # we don't use the default progress bar since that slows things down

        if tracking.best_action is None:
            vec_env = model.get_env()
            obs = vec_env.reset()
            action, _states = model.predict(obs, deterministic=True)
            best_action = action[0]
        else:
            best_action = tracking.best_action
        pred_D, pred_B = vec2admg(best_action, d, np.tril_indices(d, -1), topo_order)
        # pred_D, pred_B = vec2dag(best_action, d, np.tril_indices(d, -1)), np.zeros((d, d)) # comment above line and uncomment this line for ALIAS (2 of 7 changes to get ALIAS behaviour - search the word ALIAS in comments)
        best_bic = - tracking.best_reward * n
        pag_matrix = convert_admg_to_pag(pred_D, pred_B)
        if verbose:
            logger.info(f'\nBest BIC = {best_bic}')
            logger.info(f'Predicted ADMG (parents on columns) = \nDirected Edges:\n{pred_D.astype(int)}\nBidirected Edges:\n{pred_B}')
        # return pred_D, pred_B, pag_matrix, {'average_rewards': tracking.average_rewards, 'action_values': tracking.action_lengths}, best_bic
        return pred_D, pred_B, pag_matrix, {'average_rewards': tracking.average_rewards}, best_bic
    finally:
        vec_env.close()
        if 'model' in locals():
            del model

if __name__ == '__main__':
    start = time.perf_counter()
    logger.info('STARTED EXEC\n\n')
    seed = 32
    admg_model = 'ancestral'
    graph_gen = GraphGenerator(seed)
    D, B, X, S, bic, pag = graph_gen.get_admg(num_nodes=10, avg_degree=4, frac_directed=0.7, degree_variance=0.2, admg_model=admg_model, do_sampling=True, num_samples=1000)
    logger.info(f"BIC of TRUE graph: {bic}\n")

    est = relcadilac(X, sample_cov=S, admg_model=admg_model)
    print(f'\n\nend time taken = {(time.perf_counter() - start) / 60} mins\n\n')


