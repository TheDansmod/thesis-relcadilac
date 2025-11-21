import os
# danish: check if needed (seems to work)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
from relcadilac.utils import vec_2_bow_free_admg, vec_2_ancestral_admg, plot_rewards
from relcadilac.data_generator import GraphGenerator
from relcadilac.optim_linear_gaussian_sem import LinearGaussianSEM
from dcd.utils.admg2pag import get_graph_from_adj, admg_to_pag, get_pag_matrix

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def relcadilac(
        X,
        sample_cov,
        admg_model,   # could be either bow-free or ancestral
        steps_per_env=2000,
        n_envs=8,
        rl_params={"normalize_advantage": True, "n_epochs": 1, "device": "cuda", "verbose": 0, 'n_steps': 16, 'ent_coef': 0.05},
        verbose=1,
        random_state=0,
        **unused
    ):
    if not admg_model in ['bow-free', 'ancestral']:
        raise ValueError("admg_model must either be `bow-free` or `ancestral`")
    rl_params = deepcopy(rl_params or {})
    if admg_model == 'bow-free':
        vec2admg = vec_2_bow_free_admg
    if admg_model == 'ancestral':
        vec2admg = vec_2_ancestral_admg
    n, d = X.shape
    steps = steps_per_env * n_envs

    raw_vec_env = make_vec_env(ADMGEnv, n_envs=n_envs, env_kwargs=dict(nodes=d, X=X, sample_cov=sample_cov, vec2admg=vec2admg), vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(raw_vec_env, norm_obs=False, norm_reward=False, gamma=1.0, clip_reward=np.inf)
    tracking = TrackingCallback(total_timesteps=steps, num_samples=n, verbose=verbose)

    model = PPO("MlpPolicy", vec_env, seed=random_state, **rl_params)
    logger.info("triggering learn")
    model.learn(total_timesteps=steps, callback=tracking) # we don't use the default progress bar since that slows things down

    if tracking.best_action is None:
        vec_env = model.get_env()
        obs = vec_env.reset()
        action, _states = model.predict(obs, deterministic=True)
        best_action = action[0]
    else:
        best_action = tracking.best_action
    pred_D, pred_B = vec2admg(best_action, d, np.tril_indices(d, -1))
    best_bic = - tracking.best_reward * n
    pag_matrix = get_pag_matrix(admg_to_pag(get_graph_from_adj(pred_D, pred_B)))
    logger.info(f'\nBest BIC = {best_bic}')
    logger.info(f'Predicted ADMG (parents on columns) = \nDirected Edges:\n{pred_D.astype(int)}\nBidirected Edges:\n{pred_B}')
    plot_rewards(tracking.average_rewards)
    return pred_D, pred_B, pag_matrix

def get_specific_graph_data():
    np.random.seed(42)
    dim = 4
    size = 1000
    beta = np.array([[0, 1, 0, 0],
                     [0, 0, -1.5, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]).T

    omega = np.array([[1.2, 0, 0, 0],
                      [0, 1, 0, 0.6],
                      [0, 0, 1, 0],
                      [0, 0.6, 0, 1]])
    true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
    X = np.random.multivariate_normal([0] * dim, true_sigma, size=size)
    X = X - np.mean(X, axis=0)  # centre the data
    D = np.zeros((dim, dim))
    D[np.nonzero(beta)] = 1
    B = np.zeros((dim, dim))
    B[np.nonzero(omega)] = 1
    S = np.cov(X.T)
    try:
        np.linalg.cholesky(S)
        print('S is pos def')
    except LinAlgError:
        print('S is not pos def')
    model = LinearGaussianSEM(D, B, X, S)
    model.fit()
    bic = model.bic()
    return D, B, X, S, bic

if __name__ == '__main__':
    start = time.perf_counter()
    logger.info('STARTED EXEC\n\n')
    seed = 42
    admg_model = 'bow-free'
    # graph_gen = GraphGenerator(seed)
    # D, B, X, S, bic, pag = graph_gen.get_admg(num_nodes=10, avg_degree=4, frac_directed=0.7, degree_variance=0.1, admg_model=admg_model, do_sampling=True, num_samples=5000)
    D, B, X, S, bic = get_specific_graph_data()
    logger.info(f"BIC of TRUE graph: {bic}\n")

    est = relcadilac(X, sample_cov=S, admg_model=admg_model)
    print(f'\n\nend time taken = {(time.perf_counter() - start) / 60} mins\n\n')


