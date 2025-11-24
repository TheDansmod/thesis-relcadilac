import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ananke.graphs.admg import ADMG
from ananke.models.linear_gaussian_sem import LinearGaussianSEM as LGSem

from relcadilac.optim_linear_gaussian_sem import LinearGaussianSEM as myLGSem
from dcd.utils.admg2pag import get_graph_from_adj, admg_to_pag, get_pag_matrix

def vec_2_bow_free_admg(z, d, tril_ind, topo_order):  # still takes the argument but does nothing with it
    p = z[:d]
    diE = np.zeros((d, d))
    diE[tril_ind] = z[d:(d * (d+1)) // 2]
    D = (diE + diE.T > 0) * (p[:, None] > p[None, :])
    biE = np.zeros((d, d))
    biE[tril_ind] = z[(d * (d+1)) // 2:]
    B = (biE + biE.T > 0) * (1 - D) * (1 - D.T)
    return D, B

def vec_2_bow_free_admg_known_topo_order(z, d, tril_ind, topo_order):
    diE = np.zeros((d, d))
    diE[tril_ind] = z[: (d * (d-1)) // 2]
    D = (diE + diE.T > 0) * (topo_order[:, None] > topo_order[None, :])
    biE = np.zeros((d, d))
    biE[tril_ind] = z[(d * (d-1)) // 2:]
    B = (biE + biE.T > 0) * (1 - D) * (1 - D.T)
    return D, B

def vec_2_bow_free_admg_logits(z, d, tril_ind, topo_order):
    p = z[:d]
    m = len(tril_ind[0])
    w_di = z[d : d + m]
    w_bi = z[d + m :]

    # 3. Construct the Logic Competition
    # We create a (3, m) stack: [Null_Potentials, Dir_Potentials, Bi_Potentials]
    # Null potentials are fixed at 0.
    null_potentials = np.zeros(m)

    # Stack structure: row 0 = Null, row 1 = Directed, row 2 = Bidirected
    logits = np.vstack([null_potentials, w_di, w_bi]) # Shape (3, m)

    # 4. Determine Edge Types via Argmax
    # choices will be an array of length m with values {0, 1, 2}
    choices = np.argmax(logits, axis=0)

    # 5. Map back to Adjacency Matrices
    # Create strictly lower triangular matrices first
    D = np.zeros((d, d))
    B = np.zeros((d, d))

    D[tril_ind] = (choices == 1).astype(float)
    B[tril_ind] = (choices == 2).astype(float)

    # 6. Enforce Topological Ordering on Directed Edges
    # An edge i->j exists in the final graph ONLY if:
    #   a) The logit competition selected Directed (L_dir[i,j] == 1 or L_dir[j,i] == 1)
    #   b) The ordering condition is met.

    # Expand p to compare all pairs
    # P_diff[i, j] = p[i] - p[j]
    # P_diff = p[:, None] - p[None, :]

    # The tril_ind usually maps to indices (i, j) where i > j.
    # We populate the full matrix by adding the transpose (symmetrizing the existence decision)
    # then masking by ordering.

    # "Raw" existence based on logits (ignoring direction for a moment)
    # adj_existence_dir = D + D.T

    # Final D: Keep edge if existence is predicted AND ordering agrees
    D = (D + D.T) * (p[:, None] > p[None, :])

    # Final B: Bidirected edges are symmetric and unconstrained by ordering
    # (other than the bow-free constraint which is handled by the mutual exclusivity of argmax)
    B = B + B.T

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

def vec_2_ancestral_admg(z, d, tril_ind, topo_order):  # still takes the topo_order argument but does nothing with it
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

def vec_2_ancestral_admg_known_topo_order(z, d, tril_ind, topo_order):
    diE = np.zeros((d, d))
    diE[tril_ind] = z[: (d * (d-1)) // 2]
    D = (diE + diE.T > 0) * (topo_order[:, None] > topo_order[None, :])
    # get transitive closure matrix R
    # create adj list - cols=sources, rows=targets
    rows, cols = np.nonzero(D)
    adj_list = [[] for _ in range(d)]
    for u, v in zip(cols, rows):
        adj_list[u].append(v)
    reach = [0] * d  # bitmask of all nodes reachable from u
    for u in topo_order[::-1]:  # iterate through reverse topo order
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
    biE[tril_ind] = z[(d * (d-1)) // 2:]
    B = (biE + biE.T > 0) * (1 - R) * (1 - R.T)
    return D, B

def draw_admg(D, B, file_name, folder):
    # this takes as input the directed edge matrix (cols=sources, rows=targets)
    # and the bidirected edge matrix for and ADMG and plots it using graphviz
    # in the specified file and folder in pdf format
    d = D.shape[0]
    vertices = [f'{i}' for i in range(d)]
    di_edges = [(f'{idx[1]}', f'{idx[0]}') for idx, x in np.ndenumerate(D) if x > 0]
    bi_edges = [(f'{idx[0]}', f'{idx[1]}') for idx, x in np.ndenumerate(np.triu(B, 1)) if x > 0]
    G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
    G.draw(direction="LR").render(filename=file_name, directory=folder, format='pdf')

def draw_admg_named_vertices(D, B, vertex_names, file_name, folder):
    # same as draw_admg, but when you have names for your vertices
    d = D.shape[0]
    vertices = [vertex_names[i] for i in range(d)]
    di_edges = [(vertex_names[idx[1]], vertex_names[idx[0]]) for idx, x in np.ndenumerate(D) if x > 0]
    bi_edges = [(vertex_names[idx[0]], vertex_names[idx[1]]) for idx, x in np.ndenumerate(np.triu(B, 1)) if x > 0]
    G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
    G.draw(direction="LR").render(filename=file_name, directory=folder, format='pdf')

def get_thresholded_admg(D, B, X, S, threshold=0.05):
    # fits a linear gaussian model, then applies threshold, obtains D and B again
    d = D.shape[0]
    model = myLGSem(D, B, X, S)
    model.fit()
    D, B = np.zeros((d, d)), np.zeros((d, d))
    beta = np.where(abs(model.B_) <= threshold, 0, model.B_)
    omega = np.where(abs(model.omega_) <= threshold, 0, model.omega_)
    D[np.nonzero(beta)] = 1
    B[np.nonzero(omega)] = 1
    return D, B

def plot_rewards(rewards, save_path='diagrams/reward.png'):
    plt.plot(rewards)
    plt.xlabel("step (all envs take 1 step)")
    plt.ylabel("average reward")
    plt.title("Average rewards per step")
    plt.savefig(save_path)
    plt.close()

def get_ananke_bic(D, B, X):
    d = D.shape[0]
    vertices = [f'{i}' for i in range(d)]
    di_edges = [(f'{idx[1]}', f'{idx[0]}') for idx, x in np.ndenumerate(D) if x > 0]
    bi_edges = [(f'{idx[0]}', f'{idx[1]}') for idx, x in np.ndenumerate(np.triu(B, 1)) if x > 0]
    G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
    df_X = pd.DataFrame({f'{i}': X[:, i] for i in range(d)})
    model = LGSem(G)
    model.fit(df_X)
    return model.bic(X)


def get_bic(D, B, X, S):
    # danish: this is different, change it
    # return get_ananke_bic(D, B, X)
    model = myLGSem(D, B, X, S)
    model.fit()
    return model.bic()

def get_adj_from_ananke_graph(G):
    # takes in an ananke ADMG graph, G and returns two adjacency matrices
    num_nodes = len(G.vertices)
    dir_matrix, bidir_matrix = np.zeros((num_nodes, num_nodes)), np.zeros((num_nodes, num_nodes))
    vertex_map = {v: i for i, v in enumerate(G.vertices)}  # we don't care which vertex is mapped to which number
    for u, v in G.di_edges:  # iterate through a set of tuples of directed edges
        dir_matrix[vertex_map[v], vertex_map[u]] = 1  # if (u, v) is in di_edges then there is edge u -> v, but in our matrices if A[u, v] = 1 it means there is an edge from v to u
    for u, v in G.bi_edges:  # iterate through a set of tuples of bidirected edges - each just once
        bidir_matrix[vertex_map[v], vertex_map[u]] = 1
        bidir_matrix[vertex_map[u], vertex_map[v]] = 1
    return dir_matrix, bidir_matrix

def convert_admg_to_pag(D, B):
        return get_pag_matrix(admg_to_pag(get_graph_from_adj(D, B)))
