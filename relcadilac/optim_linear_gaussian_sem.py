import numpy as np
from numba import njit
from numba.typed import List


@njit(fastmath=True, cache=True)
def ricf_update_kernel(X, B, Omega, parent_indices, sibling_indices, n, d):
    epsilon = X - X @ B.T
    ridge_lambda = 1e-8  # for stability of OLS solve
    for var_index in range(d):
        # get epsilon_minusi (use mask for numba)
        mask_minusi = np.arange(d) != var_index
        epsilon_minusi = epsilon[:, mask_minusi]
        omega_minusi = Omega[mask_minusi, :]
        omega_minusii = omega_minusi[:, mask_minusi]
        
        # calculate Z_minusi, use solve instead of inv
        Z_minusi = (np.linalg.solve(omega_minusii, epsilon_minusi.T)).T
        
        # Y = X[:, var_index]
        # Xmat = [intercept, parents, spouses(from Z)]
        
        # Get parents and siblings from pre-computed adjacency lists
        parents = parent_indices[var_index]
        siblings = sibling_indices[var_index]
        n_parents = len(parents)
        n_siblings = len(siblings)
        n_cols = 1 + n_parents + n_siblings # 1 for intercept
        
        Y = X[:, var_index]

        Xmat = np.empty((n, n_cols), dtype=np.float64)
        Xmat[:, 0] = 1.0  # intercept
        for k in range(n_parents): # parents
            Xmat[:, 1 + k] = X[:, parents[k]]
            
        # fill spouses (pseudo-variables)
        # siblings are indexed by X, not Z_minusi
        for k in range(n_siblings): # spouses from Z
            orig_idx = siblings[k]
            # map original index to Z_minusi index
            z_idx = orig_idx if orig_idx < var_index else orig_idx - 1
            Xmat[:, 1 + n_parents + k] = Z_minusi[:, z_idx]
        # ols
        Xmat_T = Xmat.T
        gram = Xmat_T @ Xmat
        for i in range(n_cols):
            gram[i, i] += ridge_lambda
        params = np.linalg.solve(gram, Xmat_T @ Y)
        
        current_B_row = B[var_index, :].copy() # for epsilon update
        param_idx = 1
        for idx in parents:
            B[var_index, idx] = params[param_idx]
            param_idx += 1
        for idx in siblings:
            val = params[param_idx]
            Omega[var_index, idx] = val
            Omega[idx, var_index] = val
            param_idx += 1
            
        y_pred = Xmat @ params
        residuals = Y - y_pred
        scale = np.dot(residuals, residuals) / n
        
        # schur complement addition
        omega_i_minusi = Omega[var_index, mask_minusi]
        omega_minusi_i = Omega[mask_minusi, var_index]
        
        inv_prod = np.linalg.solve(omega_minusii, omega_minusi_i)
        schur_term = np.dot(omega_i_minusi, inv_prod)
        Omega[var_index, var_index] = scale + schur_term
        
        # update epsilon since we removed it out of the loop
        epsilon[:, var_index] = X[:, var_index] - X @ B[var_index, :]
        
    return B, Omega


class LinearGaussianSEM:
    def __init__(self, adj_di, adj_bi, data_matrix, sample_cov_matrix):
        self.X = data_matrix # passed in as contiguous arrays
        self.S = sample_cov_matrix # passed in as contiguous arrays
        self.n, self.d = self.X.shape
        
        self.n_params = np.sum(adj_di) + (np.sum(adj_bi) // 2) + self.d
        self._parent_index_map = List()
        self._sibling_index_map = List()
        for v in range(self.d):
            self._parent_index_map.append(np.nonzero(adj_di[v, :])[0].astype(np.int64))
            self._sibling_index_map.append(np.nonzero(adj_bi[v, :])[0].astype(np.int64))
        
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
        for _ in range(max_iters):
            # numba call - need to pass all required values
            self.B_, self.omega_ = ricf_update_kernel(self.X, self.B_, self.omega_, self._parent_index_map, self._sibling_index_map, n, d)
            new_lld = self.neg_loglikelihood()
            if np.abs(new_lld - cur_lld) < tol:
                break
            cur_lld = new_lld
        return self
