import numpy as np

from adult_income_ml.evaluation import compute_metrics


def test_compute_metrics_keys():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert "accuracy" in m
    assert "f1_macro" in m
    assert "roc_auc" in m
    assert "brier_score" in m
