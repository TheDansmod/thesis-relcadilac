import os
import logging

import numpy as np
from ananke.graphs import ADMG

from relcadilac.utils import get_transitive_closure, get_bic, convert_admg_to_pag, get_dag_bic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphGenerator:
    def __init__(self, seed = None):
        self.rng = np.random.default_rng(seed)

    def _sample_pm_uniform(self, low, high, size):
        # samples uniformly from [-high, -low] union [low, high].
        magnitudes = self.rng.uniform(low=low, high=high, size=size)
        signs = self.rng.choice([-1.0, 1.0], size=size)
        return magnitudes * signs

    def _is_connected(self, adj_dir, adj_bidir):
        # connectivity using bfs
        d = adj_dir.shape[0]
        skeleton = (adj_dir + adj_dir.T + adj_bidir) > 0
        visited = np.zeros(d, dtype=bool)
        queue = [0]
        visited[0] = True
        count = 0
        while queue:
            u = queue.pop(0)
            count += 1
            neighbors = np.where(skeleton[u, :])[0]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == d

    def sample_data_from_admg(self, adj_dir, adj_bidir, params):
        # generates data from a linear gaussian ADMG.
        d = adj_dir.shape[0]
        # generate coefficient matrix, beta
        beta = self._sample_pm_uniform(low=params['beta_low'], high=params['beta_high'], size=(d, d)) * adj_dir
        # generate error covariance matrix, omega
        mB_offdiag = self._sample_pm_uniform(low=params['omega_offdiag_low'], high=params['omega_offdiag_high'], size=(d, d)) * adj_bidir
        mB_cov = (mB_offdiag + mB_offdiag.T) / 2  # for symmetric
        row_sums = np.sum(np.abs(mB_cov), axis=1)
        diag_noise = self.rng.uniform(low=params['omega_diag_low'], high=params['omega_diag_high'], size=d)
        mB_diag = np.diag(row_sums + diag_noise)
        omega = mB_cov + mB_diag
        
        I = np.eye(d)
        inv_I_minus_beta = np.linalg.inv(I - beta)
        true_sigma = inv_I_minus_beta @ omega @ inv_I_minus_beta.T
        # ensure true_sigma is symmetric
        true_sigma = (true_sigma + true_sigma.T) / 2
        X = self.rng.multivariate_normal(mean=np.zeros(d), cov=true_sigma, size=params['num_samples'])
        if params['standardize_data']:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        elif params['center_data']:
            X = X - np.mean(X, axis=0)
        return X

    def plot_admg(self, adj_dir, adj_bidir, filename, directory):
        # plots the admg using ananke.
        # adj_dir[i, j] = 1 implies edge j -> i.
        d = adj_dir.shape[0]
        vertices = [f'X{i}' for i in range(d)]
        di_edges = [(f'X{idx[1]}', f'X{idx[0]}') for idx, x in np.ndenumerate(adj_dir) if x > 0]
        bi_edges = [(f'X{idx[0]}', f'X{idx[1]}') for idx, x in np.ndenumerate(np.triu(adj_bidir, 1)) if x > 0]
        try:
            G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
            os.makedirs(directory, exist_ok=True)
            output_path = G.draw(direction="LR").render(filename=filename, directory=directory, format='pdf')
            logger.debug(f"Graph plotted to {output_path}")
        except Exception as e:
            logger.error(f"Failed to plot ADMG: {e}")

    def get_admg(
        self, 
        num_nodes, 
        avg_degree, 
        frac_directed, 
        degree_variance = 0.0,
        admg_model = 'bow-free',  # could be either bow-free or ancestral
        plot = False, 
        plot_filename = 'bow_free_admg', 
        plot_directory = os.getcwd(), 
        do_sampling = False, 
        num_samples = 2000, 
        sampling_params = None,
        get_pag = True,
        require_connected = False,
    ):  # you will get around num_nodes * avg_degree / 2 edges in the skeleton
        # generates a random bow-free ADMG.
        # num_nodes: number of nodes in the ADMG
        # avg_degree: expected average degree of a node in the graph skeleton
        # frac_directed: in (0, 1), tells what fraction of edges in the graph are directed, 1 - frac_directed will be bidirected
        # degree_variance: if > 0 then uses beta distribution to create a less peaked degree distribution. can be [0, infty)
        # returns: pair of binary adjacency matrices for a bow-free ADMG, one for directed edges, one for bidirected edges
        if avg_degree > num_nodes - 1:
            raise ValueError(f"Avg degree {avg_degree} too high for {num_nodes} nodes.")
        if not 0 <= frac_directed <= 1:
            raise ValueError("frac_directed must be in [0, 1].")
        if not admg_model in ['bow-free', 'ancestral']:
            raise ValueError("ADMG model should be either `bow-free` or `ancestral`")

        # probs
        base_p_edge = avg_degree / (num_nodes - 1)
        
        # validate connectedness
        # danish: check this:
        degree_variance = 0.0
        while True:
            if degree_variance > 0:
                # we want a beta(a, b) distribution with mean = base_p_edge
                # mean = a / (a + b). nu = 1 / degree_variance = a + b
                # a = mean * nu, b = (1-mean) * nu
                if base_p_edge == 1:
                    base_p_edge = 0.99  # to ensure beta > 0
                nu = 1.0 / degree_variance
                alpha = base_p_edge * nu
                beta_param = (1 - base_p_edge) * nu
                p_edge_instance = self.rng.beta(alpha, beta_param)
            else:
                p_edge_instance = base_p_edge

            p_dir = p_edge_instance * frac_directed
            p_bidir = p_edge_instance * (1 - frac_directed)
            p_none = 1.0 - (p_dir + p_bidir)
            if p_none < 0: 
                p_none = 0.3 # default for safety

            # conditional probability for bidirected edge given no directed edge exists - for ancestral graphs
            if p_none + p_bidir > 0:
                p_bidir_cond = p_bidir / (p_none + p_bidir)
            else:
                p_bidir_cond = 0.0

            probs = np.array([p_none, p_dir, p_bidir])
            probs /= probs.sum() # ensure sum exactly 1.0

            # adj matrices
            adj_dir = np.zeros((num_nodes, num_nodes))
            adj_bidir = np.zeros((num_nodes, num_nodes))
            
            # generate only for upper triangle - for directed it ensures acyclicity, and for bi-directed it will be mirrored
            iu_indices = np.triu_indices(num_nodes, k=1)
            num_pairs = len(iu_indices[0])
            
            # 0 is no edge, 1 is directed edge, 2 is bidirected edge
            choices = self.rng.choice([0, 1, 2], size=num_pairs, p=probs)
            
            # directed edges
            if admg_model == 'bow-free':
                idx_dir = np.where(choices == 1)[0]
            if admg_model == 'ancestral':
                idx_dir = np.where(self.rng.random(num_pairs) < p_dir)[0]
            u_dir = iu_indices[0][idx_dir]
            v_dir = iu_indices[1][idx_dir]
            adj_dir[v_dir, u_dir] = 1 
            
            # bidirected edges - different for ancestral and bow-free
            if admg_model == 'bow-free':
                idx_bidir = np.where(choices == 2)[0]
                u_bidir = iu_indices[0][idx_bidir]
                v_bidir = iu_indices[1][idx_bidir]
                adj_bidir[u_bidir, v_bidir] = 1
                adj_bidir[v_bidir, u_bidir] = 1
            if admg_model == 'ancestral':
                dag_tc = get_transitive_closure(num_nodes, adj_dir)
                bidir_random = self.rng.random(num_pairs)
                for k in range(num_pairs):
                    u, v = iu_indices[0][k], iu_indices[1][k]
                    is_ancestral_link = dag_tc[u, v] or dag_tc[v, u]
                    if not is_ancestral_link:
                        if bidir_random[k] < p_bidir_cond:
                            adj_bidir[u, v] = 1
                            adj_bidir[v, u] = 1
            
            # permute nodes
            perm = self.rng.permutation(num_nodes)
            adj_dir = adj_dir[perm, :][:, perm]
            adj_bidir = adj_bidir[perm, :][:, perm]
            
            # ensure the number of actual edges and expected don't differ by more than 5 and the fraction of directed to bidirected edges is also close (no more than difference of 0.1)
            actual_num_edges = np.sum(adj_dir) + (np.sum(adj_bidir) // 2)
            expected_num_edges = avg_degree * num_nodes / 2
            actual_frac_dir = np.sum(adj_dir) / actual_num_edges
            if abs(expected_num_edges - actual_num_edges) > 5:
                continue
            if abs(actual_frac_dir - frac_directed) > 0.1:
                continue

            # if not connected repeat
            if (not require_connected) or self._is_connected(adj_dir, adj_bidir):
                break

        # log num nodes and edges
        n_dir = np.sum(adj_dir)
        n_bidir = np.sum(np.triu(adj_bidir, 1))
        logger.debug(f"Generated {admg_model} ADMG. Num Nodes: {num_nodes}. Num Directed Edges: {n_dir}. Num Bidirected Edges: {n_bidir}.")

        # plotting
        if plot:
            self.plot_admg(adj_dir, adj_bidir, plot_filename, plot_directory)
            
        # Sampling
        samples, data_cov_matrix, bic, pag_matrix = None, None, None, None
        if do_sampling:
            if sampling_params is None:
                sampling_params = {}
            defaults = {'beta_low': 0.5, 'beta_high': 2.0, 'omega_offdiag_low': 0.4, 'omega_offdiag_high': 0.7, 'omega_diag_low': 0.7, 'omega_diag_high': 1.2, 'num_samples': num_samples, 'standardize_data': False, 'center_data': True}
            final_params = {**defaults, **sampling_params}
            samples = self.sample_data_from_admg(adj_dir, adj_bidir, final_params)
            data_cov_matrix = np.cov(samples.T)
            bic = get_bic(adj_dir, adj_bidir, samples, data_cov_matrix)
        # pag
        if get_pag:
            pag_matrix = convert_admg_to_pag(adj_dir, adj_bidir)
        return adj_dir, adj_bidir, samples, data_cov_matrix, bic, pag_matrix

    def get_lin_gauss_ev_dag(self, num_nodes, avg_degree, plot=False, plot_filename='lin_gauss_ev_dag', plot_directory=os.getcwd(), do_sampling=False, num_samples=2000, sampling_params=None):
        if avg_degree > num_nodes - 1:
            raise ValueError(f"Avg degree {avg_degree} too high for {num_nodes} nodes.")
        p_edge = avg_degree / (num_nodes - 1)
        adj = np.zeros((num_nodes, num_nodes))
        triu_ind = np.triu_indices(num_nodes, k=1)
        num_vals = len(triu_ind[0])
        rnd = self.rng.uniform(low=0.0, high=1.0, size=num_vals)
        edge_idx = np.where(rnd < p_edge)[0]
        u = triu_ind[0][edge_idx]
        v = triu_ind[1][edge_idx]
        adj[v, u] = 1 
        perm = self.rng.permutation(num_nodes)
        adj = adj[perm, :][:, perm]
        if plot:
            self.plot_admg(adj, np.zeros((num_nodes, num_nodes)), plot_filename, plot_directory)
        X, bic = None, None
        if do_sampling:
            if sampling_params is None:
                sampling_params = {}
            defaults = {'beta_low': 0.5, 'beta_high': 2.0, 'num_samples': num_samples, 'standardize_data': False, 'center_data': True}
            params = {**defaults, **sampling_params}
            W = self._sample_pm_uniform(low=params['beta_low'], high=params['beta_high'], size=(num_nodes, num_nodes)) * adj
            E = self.rng.standard_normal(size=(num_nodes, params['num_samples']))
            X = np.linalg.solve((np.eye(num_nodes) - W), E).T
            if params['standardize_data']:
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            elif params['center_data']:
                X = X - np.mean(X, axis=0)
            bic = get_dag_bic(adj, X)
        return adj, X, bic

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    SEED = 22
    generator = GraphGenerator(seed=SEED)
    
    num_nodes = 7
    avg_degree = 2
    frac_directed = 0.6
    
    adj_d, adj_b, X, S, bic, pag = generator.get_admg(
        num_nodes=num_nodes, 
        avg_degree=avg_degree, 
        frac_directed=frac_directed,
        degree_variance=0.1,
        plot=True,
        do_sampling=False,
    )
    D, X, bic = generator.get_lin_gauss_ev_dag(num_nodes=10, avg_degree=4, plot=True, plot_filename='test_dag_can_del', plot_directory='diagrams/', do_sampling=True, num_samples=2000)
