"""Subgroup fairness metrics."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix

from adult_income_ml.utils import load_config


def _rates(cm: list[list[int]]) -> dict[str, float]:
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"fpr": fpr, "fnr": fnr, "precision": prec, "recall": rec, "support": tn + fp + fn + tp}


def subgroup_metrics(
    df: pd.DataFrame,
    y_true,
    y_pred,
    attribute: str,
    min_size: int | None = None,
) -> pd.DataFrame:
    cfg = load_config()
    min_size = min_size or cfg["fairness"]["min_subgroup_size"]
    rows = []
    for group, sub in df.groupby(attribute, dropna=False):
        if len(sub) < min_size:
            continue
        yt = y_true.loc[sub.index] if hasattr(y_true, "loc") else y_true[sub.index]
        yp = y_pred.loc[sub.index] if hasattr(y_pred, "loc") else y_pred[sub.index]
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        rates = _rates(cm.tolist())
        rows.append({"group": str(group), "attribute": attribute, **rates})
    return pd.DataFrame(rows)


def intersectional_metrics(
    df: pd.DataFrame,
    y_true,
    y_pred,
    attrs: list[str],
    min_size: int | None = None,
) -> pd.DataFrame:
    cfg = load_config()
    min_size = min_size or cfg["fairness"]["min_subgroup_size"]
    work = df.copy()
    work["_inter"] = work[attrs[0]].astype(str)
    for a in attrs[1:]:
        work["_inter"] = work["_inter"] + "|" + work[a].astype(str)
    return subgroup_metrics(work, y_true, y_pred, "_inter", min_size)
