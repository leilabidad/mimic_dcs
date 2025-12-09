"""
Evaluation metrics: classification, calibration, consistency metrics.
"""

from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, accuracy_score
import numpy as np

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob).reshape(-1) >= threshold).astype(int)
    y_true = np.array(y_true).reshape(-1).astype(int)
    out = {}
    try:
        out['auc'] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out['auc'] = None
    out['f1'] = float(f1_score(y_true, y_pred)) if y_true.sum() > 0 else None
    out['acc'] = float(accuracy_score(y_true, y_pred))
    out['brier'] = float(brier_score_loss(y_true, y_prob))
    return out

def consistency_metric(rf_list):
    """
    Simple consistency metric across repeated runs:
    compute std dev across multiple Rf evaluations per sample.
    Lower std => more consistent.
    rf_list: list of lists, shape [n_runs][n_samples]
    """
    arr = np.array(rf_list)  # [n_runs, n_samples]
    stds = np.std(arr, axis=0)
    return {'rf_std_mean': float(np.mean(stds)), 'rf_std_median': float(np.median(stds))}
