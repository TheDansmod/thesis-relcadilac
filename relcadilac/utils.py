import numpy as np

def vec_2_bow_free_admg(z, d, tril_ind):
    p = z[:d]
    diE = np.zeros((d, d))
    diE[tril_ind] = z[d:(d * (d+1)) // 2]
    D = (diE + diE.T > 0) * (p[:, None] > p[None, :])
    biE = np.zeros((d, d))
    biE[tril_ind] = z[(d * (d+1)) // 2:]
    B = (biE + biE.T > 0) * (1 - D) * (1 - D.T)
    return D, B

def get_transitive_closure(num_nodes, adj):
    # adj[i, j] = 1 ==> j -> i edge exists
    # same with R
    # a more efficient version is used for vec2admg functions
    num_iters = int(np.ceil(np.log2(num_nodes))) + 1
    R = adj.astype(bool)
    for _ in range(num_iters):
        R_next = R | (R @ R)
        if np.array_equal(R_next, R):
            break
        R = R_next
    return R.astype(float).T


def vec_2_ancestral_admg(z, d, tril_ind):
    p = z[:d]
    diE = np.zeros((d, d))
    diE[tril_ind] = z[d:(d * (d+1)) // 2]
    D = (diE + diE.T > 0) * (p[:, None] > p[None, :])
    # get transitive closure matrix R
    # create adj list - cols=sources, rows=targets
    rows, cols = np.nonzero(D)
    adj_list = [[] for _ in range(d)]
    for u, v in zip(cols, rows):
        adj_list[u].append(v)
    reach = [0] * d  # bitmask of all nodes reachable from u
    for u in np.argsort(p)[::-1]:  # iterate through reverse topo order
        r_u = 0
        for v in adj_list[u]:
            # u reaches v (1 << v) and everything v reaches (reach[v])
            r_u |= (1 << v) | reach[v]
        reach[u] = r_u
    R = np.zeros((d, d), dtype=float)
    for i in range(d):
        mask = reach[i]
        for j in range(d):
            if (mask >> j) & 1:
                R[j, i] = 1.0
    biE = np.zeros((d, d))
    biE[tril_ind] = z[(d * (d+1)) // 2:]
    B = (biE + biE.T > 0) * (1 - R) * (1 - R.T)
    return D, B
