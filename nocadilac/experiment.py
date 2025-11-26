import numpy as np
import networkx as nx
from data_gen import DataGen
from model import ConfoundedAE as CAE, train
from util import metrics, conf_eval, acyclify
from pathlib import Path


# Synthetic data: Sec 5.1
def synthetic(N=1000):
    for M in [10, 25, 50]:
        for gen in ['lin', 'quad', 'cub', 'log', 'exp', 'sin']:
            dg  = DataGen(N, M, 0.3, gen=gen)
            dg.net()
            x = dg.data()
            conf = dg.conf_ind

            model = CAE(M, 1)
            best_A, best_B = train(model, x)
            A = acyclify(best_A.numpy())
            best_B = best_B.numpy()

            G_true = dg.G
            G_guess = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
            prec, rec, shd, sid, sid_rev = metrics(G_guess, G_true)

            prec_c, rec_c = conf_eval(best_B, conf)

            with open(f'res/synth/noc-{M}-{gen}.csv', 'a') as f:
                f.write(f'{prec}, {rec}, {shd}, {sid}, {sid_rev}, {prec_c}, {rec_c}')
            break
        break

# REGED: Sec. 5.2
def reged():
    # REGED graph and data
    G = nx.from_scipy_sparse_matrix(loadmat('data/reged/GS.mat')['DAG'])
    data = loadmat('data/reged/data_obs.mat')['data_obs']

    pars = []
    for i in G.nodes:
        if G.out_degree(i) >= 3:
            pars.append(i)

    # Generate graphs around the children of nodes with outdegree 3+
    gs = []
    for i in pars:
        g = []
        nodes = []
        for c in G.successors(i):
            preds = list(G.predecessors(c))
            succ = list(G.successors(c))
            sibs = []
            for s in succ:
                sibs += list(G.predecessors(s))
            mb = preds + succ + sibs
            nodes += mb
        nodes = set(nodes)
        g = G.subgraph(nodes)
        gs.append(g)

    gs = [g for g in gs if 10 <= len(g.nodes) <= 100]

    for i, g in enumerate(gs):
        nodes = list(g.nodes)
        x = data.iloc[:, nodes]

        model = CAE(len(nodes), 1)
        best_A, best_B = train(model, x)
        A = acyclify(best_A.numpy())
        best_B = best_B.numpy()
        G_true = dg.G
        G_guess = nx.from_numpy_matrix(A, create_using=nx.DiGraph)

        prec, rec, shd, sid, sid_rev = metrics(G_guess, G_true)
        prec_c, rec_c = conf_eval(best_B, conf)

        with open(f'res/reged/noc-reged.csv', 'a') as f:
            f.write(f'{prec}, {rec}, {shd}, {sid}, {sid_rev}, {prec_c}, {rec_c}')

# Sachs: Sec. 5.3
def sachs():
    # Sachs graph an ddata
    G = nx.from_scipy_sparse_matrix(loadmat('data/sachs/dag.mat')['DAG'])
    x = loadmat('data/sachs/data.mat')['data']
    x = x[:, 1:]

    nodes = list(G.nodes())
    model = CAE(len(nodes), 1)
    best_A, best_B = train(model, x)
    A = acyclify(best_A.numpy())
    best_B = best_B.numpy()

    G_true = dg.G
    G_guess = nx.from_numpy_matrix(A, create_using=nx.DiGraph)

    with open(f'res/sachs/noc-sachs.csv', 'a') as f:
        f.write(f'{prec}, {rec}, {shd}, {sid}, {sid_rev}, {prec_c}, {rec_c}')

# danish function below:
def test_working():
    N = 1000  # number of samples
    M = 5  # number of nodes
    gen = 'lin'  # type of function
    dg  = DataGen(N, M, frac_confounded=0.3, gen=gen)
    print('\n\ndanish: generated\n\n')
    dg.net()
    x = dg.data()  # x is a numpy nd-array
    print(f'\n\ndanish: {x.shape=}')
    conf = dg.conf_ind
    print(f'confounded indices = {conf}')

    model = CAE(M, 1)
    threshold = 0.05  # threshold code danish has added - you don't need to .numpy the A and B since I am already doing it in model
    best_A, best_B = train(model, x)  # best_A, best_B are of type backend.Variable - they might be tensorflow variables
    print('best_A\n', best_A)
    best_A[np.abs(best_A) < threshold] = 0
    A = acyclify(best_A)
    best_B = best_B  # A (M x M shape), best_B (1, M shape) are numpy nd-arrays - A looks to be the adjacency matrix (but it has non-binary entries), and B is the co-efficient of the latent variables: X = A'X + B'Z + eps, where B and Z are 1-dimensional
    G_true = dg.G
    G_guess = nx.from_numpy_array(A, create_using=nx.DiGraph)
    print("A")
    print(A)
    print()
    print("danish: G_true")
    print(G_true.edges())
    print("danish: G_guess")
    print(G_guess.edges())

if __name__ == '__main__':
    # Path('res/synth').mkdir(parents=True, exist_ok=True)
    # Path('res/reged').mkdir(parents=True, exist_ok=True)
    # Path('res/sachs').mkdir(parents=True, exist_ok=True)

    # synthetic()
    # reged()
    # sachs()
    test_working()

