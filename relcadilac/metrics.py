import numpy as np

def get_admg_metrics(true_admg, pred_admg, num_decimal_places=4):
    # true_admg and pred_admg are tuples containing 2 binary adjacency matrices each
    # first adj matrix (D) is for directed edges and D[i, j] = 1 means there is an edge j -> i
    # second adj matrix (B) is for bidirected edges
    # calculating tpr, fdr, f1, shd for admg; tpr, fdr, f1 for skeleton
    true_D, true_B = true_admg
    pred_D, pred_B = pred_admg
    if not np.array_equal(pred_B, pred_B.T):
        raise ValueError("the predicted binary adjacency matrix for bidirected edges (pred_B) must be symmetric.")
    if not np.array_equal(true_B, true_B.T):
        raise ValueError("the ground truth binary adjacency matrix for bidirected edges (true_B) must be symmetric.")
    fnz_true_D, fnz_pred_D, fnz_true_B, fnz_pred_B = np.flatnonzero(true_D), np.flatnonzero(pred_D), np.flatnonzero(true_B), np.flatnonzero(pred_B)
    true_pos = np.intersect1d(fnz_true_D, fnz_pred_D, assume_unique=True).shape[0]
    true_pos += np.intersect1d(fnz_true_B, fnz_pred_B, assume_unique=True).shape[0] // 2  # count bidirected edge only once
    false_pos = np.setdiff1d(fnz_pred_D, fnz_true_D, assume_unique=True).shape[0]
    false_pos += np.setdiff1d(fnz_pred_B, fnz_true_B, assume_unique=True).shape[0] // 2  # count bidirected edge only once
    # tpr (true positive rate) = tp / (tp + fn) = tp / total_positives = recall
    recall = tpr = true_pos / max(np.sum(true_D) + (np.sum(true_B) // 2), 1)
    # fdr (false discovery rate) = fp / (tp + fp)
    # precision = tp / (tp + fp)
    if true_pos + false_pos <= 0:
        fdr, precision = 1.0, 0.0
    else:
        fdr = false_pos / (true_pos + false_pos)
        precision = true_pos / (true_pos + false_pos)
    # F1 score = 2 / ((1/recall) + (1/precision))
    if precision + recall <= 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # SHD - double count for reverse edge
    shd_d = np.sum(np.abs(true_D - pred_D))
    shd_b = np.sum(np.abs(np.triu(true_B, k=1) - np.triu(pred_B, k=1)))
    shd = shd_d + shd_b
    metrics = {'admg': {'tpr': tpr, 'fdr': fdr, 'f1': f1, 'shd': shd}}
    # skeleton
    tpr, fdr, f1 = get_tpr_fdr_f1(true_D + true_B, pred_D + pred_B)
    metrics['skeleton'] = {'tpr': tpr, 'fdr': fdr, 'f1': f1}
    return metrics

def get_tpr_fdr_f1(mat_true, mat_pred):
    # t: true matrix, p: predicted matrix
    fnz_true, fnz_pred = np.flatnonzero(mat_true), np.flatnonzero(mat_pred)
    true_pos = np.intersect1d(fnz_true, fnz_pred, assume_unique=True).shape[0]
    false_pos = np.setdiff1d(fnz_pred, fnz_true, assume_unique=True).shape[0]
    false_neg = np.setdiff1d(fnz_true, fnz_pred, assume_unique=True).shape[0]
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)  # denominator can't be 0
    if (true_pos + false_neg) <= 0:
        true_pos_rate = 0
    else:
        true_pos_rate = true_pos / (true_pos + false_neg)
    if (true_pos + false_pos) <= 0:
        false_disc_rate = 1
    else:
        false_disc_rate = false_pos / (true_pos + false_pos)
    return true_pos_rate, false_disc_rate, f1

def get_pag_metrics(pag_true, pag_pred, num_decimal_places=3):
    # pag_true and pag_pred are matrices representing partial ancestral graphs
    # pag[i, j] = 0 means no edge between i and j, else there is an edge
    # pag[i, j] = 1 means on edge between i and j, j has a circle
    # pag[i, j] = 2 means on edge between i and j, j has a head / arrow
    # pag[i, j] = 3 means on edge between i and j, j has a tail
    # calculating F1, tpr, fdr for skeleton, circle, head, tail
    metrics = {'skeleton': dict(), 'circle': dict(), 'head': dict(), 'tail': dict()}
    d = pag_true.shape[0]
    #### skeleton ####
    tpr, fdr, f1 = get_tpr_fdr_f1(pag_true, pag_pred)
    metrics['skeleton']['f1'] = f1
    metrics['skeleton']['tpr'] = tpr
    metrics['skeleton']['fdr'] = fdr
    #### rest ####
    for catg, val in zip(['circle', 'head', 'tail'], [1, 2, 3]):
        matrix_true, matrix_pred = np.zeros((d, d)), np.zeros((d, d))
        matrix_true[np.where(pag_true == val)] = 1
        matrix_pred[np.where(pag_pred == val)] = 1
        tpr, fdr, f1 = get_tpr_fdr_f1(matrix_true, matrix_pred)
        metrics[catg]['tpr'] = tpr
        metrics[catg]['fdr'] = fdr
        metrics[catg]['f1'] = f1
    return metrics
