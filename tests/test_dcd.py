import time

import pandas as pd

from relcadilac.data_generator import GraphGenerator
from relcadilac.metrics import get_admg_metrics, get_pag_metrics
from dcd.admg_discovery import Discovery

def run_dcd():
    params = dict()
    D, B, X, S, bic, pag = generator.get_admg(num_nodes=10, avg_degree=4, frac_directed=0.7, degree_variance=0.1, admg_model='ancestral', plot=False, do_sampling=True, num_samples=500)
    df_X = pd.DataFrame({f'{i}': X[:, i] for i in range(10)})
    learn = Discovery()  # using all default parameters
    start = time.perf_counter()
    pred_D, pred_B, pred_pag = learn.discover_admg(df_X, admg_class='ancestral', local=False, verbose=True, num_restarts=1)
    params['dcd_time_sec'] = time.perf_counter() - start
    params['dcd_admg_metrics'] = get_admg_metrics((D, B), (pred_D, pred_B))
    params['dcd_pag_metrics'] = get_pag_metrics(pag, pred_pag)
    print(params)

if __name__ == '__main__':
    generator = GraphGenerator(21)
    run_dcd()
