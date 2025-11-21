import numpy as np
import pandas as pd
import gymnasium as gym
from lru import LRU
from gymnasium import spaces

from relcadilac.optim_linear_gaussian_sem import LinearGaussianSEM

class ADMGEnv(gym.Env):
    def __init__(self, nodes, X, sample_cov, vec2admg):
        super().__init__()
        self.data = np.ascontiguousarray(X)
        self.sample_cov = np.ascontiguousarray(sample_cov)
        self.vec2admg = vec2admg
        self.n, self.d = X.shape
        self.nodes = nodes
        action_shape = (nodes ** 2,)
        self.action_space = spaces.Box(-10, 10, action_shape)
        self.observation_space = spaces.Discrete(1)
        self.tril_indices = np.tril_indices(nodes, -1)
        self._cache = LRU(50_000)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._obs = np.array(0)
        return self._obs, {}

    def evaluate(self, adj_matrices):
        D, B = adj_matrices
        key = (D.tobytes(), B.tobytes())
        if key in self._cache:
            return self._cache[key]
        # D is the binary adjacency matrix for the directed edges
        # B is the symmetric binary adjacency matrix for the bidirected edges
        model = LinearGaussianSEM(D, B, self.data, self.sample_cov)
        model.fit()
        bic = model.bic()
        reward = - bic / self.n
        self._cache[key] = reward
        return reward # we want to minimise the BIC, but this is reward, so we are maximising -bic

    def step(self, action):
        admg = self.vec2admg(action, self.d, self.tril_indices)
        self._obs = np.array(0)
        reward = self.evaluate(admg)
        terminated = True
        truncated = False
        info = {"action_vector": action}
        return self._obs, reward, terminated, truncated, info
    
