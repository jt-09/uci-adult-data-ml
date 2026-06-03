"""Probability calibration."""

from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from adult_income_ml.evaluation import compute_metrics


def calibrate_model(estimator, X_train, y_train, method: str = "isotonic", cv: int = 3):
    return CalibratedClassifierCV(estimator, method=method, cv=cv)


def calibration_metrics(y_true, y_proba) -> dict:
    return {
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        **{k: v for k, v in compute_metrics(y_true, (y_proba >= 0.5).astype(int), y_proba).items() if k == "brier_score" or True},
    }


def calibration_summary_table(y_true, y_proba) -> pd.DataFrame:
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    return pd.DataFrame({"mean_predicted": prob_pred, "fraction_positive": prob_true})
