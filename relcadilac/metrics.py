import numpy as np

def get_admg_metrics(true_admg, pred_admg, num_decimal_places=4):
    true_D, true_B = true_admg
    pred_D, pred_B = pred_admg
    fnz_true_D, fnz_pred_D, fnz_true_B, fnz_pred_B = np.flatnonzero(true_D), np.flatnonzero(pred_D), np.flatnonzero(true_B), np.flatnonzero(pred_B)
    # true positive
    true_pos = np.intersect1d(fnz_true_D, fnz_pred_D, assume_unique=True).shape[0]
    true_pos += np.intersect1d(fnz_true_B, fnz_pred_B, assume_unique=True).shape[0] // 2  # count bidirected edge only once
    # false positive
    false_pos = np.setdiff1d(fnz_pred_D, fnz_true_D, assume_unique=True).shape[0]
    false_pos += np.setdiff1d(fnz_pred_B, fnz_true_B, assume_unique=True).shape[0] // 2  # count bidirected edge only once
    # tpr (true positive rate) = tp / (tp + fn) = tp / total_positives = recall
    recall = tpr = true_pos / max(np.sum(true_D) + (np.sum(true_B) // 2), 1)
    # fdr (false discovery rate) = fp / (tp + fp)
    fdr = false_pos / (true_pos + false_pos)
    # precision = tp / (tp + fp)
    precision = true_pos / (true_pos + false_pos)
    # F1 score = 2 / ((1/recall) + (1/precision))
    f1 = 2 * precision * recall / (precision + recall)
    # SHD
    shd_d = np.sum(np.abs(true_D - pred_D))
    shd_b = np.sum(np.abs(np.triu(true_B, k=1) - np.triu(pred_B, k=1)))
    shd = shd_d + shd_b
