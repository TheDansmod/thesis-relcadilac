import numpy as np
import pandas as pd
import gymnasium as gym
from lru import LRU
from gymnasium import spaces

from relcadilac.utils import get_bic
# from relcadilac.utils import get_dag_bic, vec2dag # uncomment current line and comment previous line for ALIAS (3 of 7 changes to get ALIAS behaviour - search the word ALIAS in comments)

class ADMGEnv(gym.Env):
    def __init__(self, nodes, X, sample_cov, vec2admg, topo_order=None):
        super().__init__()
        self.data = np.ascontiguousarray(X)
        self.sample_cov = np.ascontiguousarray(sample_cov)
        self.vec2admg = vec2admg
        self.n, self.d = X.shape
        self.nodes = nodes
        action_shape = (nodes ** 2,) if topo_order is None else (nodes * (nodes - 1),)
        # action_shape = ((nodes * (nodes + 1)) // 2, )  # uncomment current line and comment previous line for ALIAS (4 of 7 changes to get ALIAS behaviour - search the word ALIAS in comments)
        self.action_space = spaces.Box(-10, 10, action_shape)
        self.observation_space = spaces.Discrete(1)
        self.tril_indices = np.tril_indices(nodes, -1)
        self._cache = LRU(50_000)
        self.topo_order = topo_order

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._obs = np.array(0)
        return self._obs, {}

    def evaluate(self, adj_matrices):
        D, B = adj_matrices
        key = (D.tobytes(), B.tobytes())
        # key = adj_matrices.tobytes()  # uncomment current line and comment previous 2 lines for ALIAS (5 of 7 changes to get ALIAS behaviour - search the word ALIAS in comments)
        if key in self._cache:
            return self._cache[key]
        # D is the binary adjacency matrix for the directed edges
        # B is the symmetric binary adjacency matrix for the bidirected edges
        reward = - get_bic(D, B, self.data, self.sample_cov) / self.n
        # reward = - get_dag_bic(adj_matrices, self.data) / self.n  # uncomment current line and comment previous line for ALIAS (6 of 7 changes to get ALIAS behaviour - search the word ALIAS in comments)
        self._cache[key] = reward
        return reward # we want to minimise the BIC, but this is reward, so we are maximising -bic

    def step(self, action):
        admg = self.vec2admg(action, self.d, self.tril_indices, self.topo_order)
        # admg = vec2dag(action, self.d, self.tril_indices)  # uncomment current line and comment previous line for ALIAS (7 of 7 changes to get ALIAS behaviour - search the word ALIAS in comments)
        self._obs = np.array(0)
        reward = self.evaluate(admg)
        terminated = True
        truncated = False
        info = {"action_vector": action}
        return self._obs, reward, terminated, truncated, info
    
