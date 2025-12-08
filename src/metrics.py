# src/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, brier_score_loss

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float('nan')

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)

    results = {
        'auc': safe_auc(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'brier': brier_score_loss(y_true, y_prob)
    }
    return results
