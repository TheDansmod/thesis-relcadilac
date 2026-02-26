import numpy as np
from numba import njit
from numba.typed import List
from scipy.sparse import csr_matrix
from collections import defaultdict
from scipy.sparse.csgraph import connected_components


@njit(fastmath=True, cache=True)
def ricf_update_kernel(X, B, Omega, parent_indices, sibling_indices, active_vertices, vertex_districts, n, d):
    epsilon = X - X @ B.T
    for vi in range(len(active_vertices)):
        var_index = active_vertices[vi]
        mask_disi = np.zeros(d, dtype=np.bool_)
        mask_disi[vertex_districts[var_index]] = True
        n_disi = len(vertex_districts[var_index])
        # get epsilon_disi (use mask for numba)
        epsilon_disi = epsilon[:, mask_disi]
        omega_disi = Omega[mask_disi, :]
        omega_disi_disi = omega_disi[:, mask_disi]
        
        # calculate Z_disi, use solve instead of inv
        if n_disi > 0:
            Z_disi = (np.linalg.solve(omega_disi_disi, epsilon_disi.T)).T
        else:
            Z_disi = np.empty((n, 0), dtype=np.float64)
        
        # get parents and siblings from pre-computed adjacency lists
        parents = parent_indices[var_index]
        siblings = sibling_indices[var_index]
        n_parents = len(parents)
        n_siblings = len(siblings)
        n_cols = n_parents + n_siblings # no intercept (+1) since we assume data is centered
        
        X_i = X[:, var_index]

        Xmat = np.empty((n, n_cols), dtype=np.float64)
        for k in range(n_parents): # parents
            Xmat[:, k] = X[:, parents[k]]
            
        # fill spouses (pseudo-variables)
        # siblings are indexed by X, not Z_disi
        for k in range(n_siblings): # spouses from Z
            orig_idx = siblings[k]
            # map original index to Z_disi index
            z_idx = -1
            for index, val in enumerate(vertex_districts[var_index]):
                if val == orig_idx:
                    z_idx = index
                    break
            Xmat[:, n_parents + k] = Z_disi[:, z_idx]
        # ols; we need ssr since w_ii_-i = ssr / n
        if n_cols > 0:
            params, _, _, _ = np.linalg.lstsq(Xmat, X_i)
            ssr = np.sum((X_i - Xmat @ params) ** 2)
        else:
            params = np.empty(0, dtype=np.float64)
            ssr = np.sum(X_i ** 2)
        
        param_idx = 0
        # beta_ij, j in pa(i)
        for idx in parents:
            B[var_index, idx] = params[param_idx]
            param_idx += 1
        # omega_ik, k in sp(i)
        for idx in siblings:
            val = params[param_idx]
            Omega[var_index, idx] = val
            Omega[idx, var_index] = val
            param_idx += 1
            
        # schur complement addition
        omega_i_disi = Omega[var_index, mask_disi]
        omega_disi_i = Omega[mask_disi, var_index]
        
        if n_disi > 0:
            inv_prod = np.linalg.solve(omega_disi_disi, omega_disi_i)
            schur_term = np.dot(omega_i_disi, inv_prod)
        else:
            schur_term = 0.0
        Omega[var_index, var_index] = (ssr / n) + schur_term
        
        # update epsilon since we removed it out of the loop
        epsilon[:, var_index] = X[:, var_index] - X @ B[var_index, :]
        
    return B, Omega

def get_vertex_districts(adj_bi):
    graph_sparse = csr_matrix(adj_bi)
    n_components, labels = connected_components(csgraph=graph_sparse, directed=False, return_labels=True)

    component_members = defaultdict(list)
    for idx, lbl in enumerate(labels):
        component_members[lbl].append(idx)

    districts = List()
    for idx in range(len(labels)):
        members = [v for v in component_members[labels[idx]] if v != idx]
        districts.append(np.array(members, dtype=np.int64))
    return districts

class LinearGaussianSEM:
    def __init__(self, adj_di, adj_bi, data_matrix, sample_cov_matrix):
        # data matrix must be centered before being passed in
        # adj_di, adj_bi must represent acyclic directed mixed graphs (ADMGs)
        # adj_di[i, j] = 1 <=> j->i in graph, else 0
        # adj_bi[i, j] = 1 <=> j<->i in graph, else 0 (can be assumed to be symmetric)
        # sample_cov_matrix must be X.T @ X / n, not / (n-1) which is done by np.cov
        self.X = data_matrix # expected to be passed in as contiguous arrays
        self.S = sample_cov_matrix # expected to be passed in as contiguous arrays
        self.n, self.d = self.X.shape
        self.n_params = np.sum(adj_di) + (np.sum(adj_bi) // 2) + self.d

        self._parent_index_map = List()
        self._sibling_index_map = List()
        for v in range(self.d):
            self._parent_index_map.append(np.nonzero(adj_di[v, :])[0].astype(np.int64))
            self._sibling_index_map.append(np.nonzero(adj_bi[v, :])[0].astype(np.int64))

        self._all_vertices = np.arange(self.d, dtype=np.int64)
        self._spouse_vertices = np.array([v for v in range(self.d) if len(self._sibling_index_map[v]) > 0], dtype=np.int64)
        self._vertex_districts = get_vertex_districts(adj_bi)
        
        self.B_ = None  # direct edge coefficients
        self.omega_ = None  # correlation of errors

    def neg_loglikelihood(self):
        inv_eye_minus_B_ = np.linalg.inv(np.eye(self.d) - self.B_)
        sigma = inv_eye_minus_B_ @ self.omega_ @ inv_eye_minus_B_.T
        
        # using solve in place of np.dot(np.linalg.inv(sigma), self.S) - solve is more stable and less expensive than O(d^3) inverse and dot
        # using slogdet instead of log of det since that is more stable
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0: return np.inf
        val = logdet + np.trace(np.linalg.solve(sigma, self.S))
        return (self.n / 2) * val

    def bic(self):
        return 2 * self.neg_loglikelihood() + np.log(self.n) * self.n_params

    def fit(self, tol=1e-6, max_iters=100):
        # initialize B and omega
        n, d = self.n, self.d
        self.B_, self.omega_ = np.zeros((d, d)), np.eye(d)
        cur_lld = self.neg_loglikelihood()
        for iter_num in range(max_iters):
            # numba call - need to pass all required values
            active = self._all_vertices if iter_num == 0 else self._spouse_vertices
            self.B_, self.omega_ = ricf_update_kernel(self.X, self.B_, self.omega_, self._parent_index_map, self._sibling_index_map, active, self._vertex_districts, n, d)
            new_lld = self.neg_loglikelihood()
            assert new_lld <= (cur_lld + 1e-12)
            if (np.abs(new_lld - cur_lld) / (np.abs(cur_lld) + 1e-12)) < tol:
                break
            cur_lld = new_lld
        return self
