"""Model registry and PyTorch embedding MLP."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from adult_income_ml.utils import load_config


def get_model(name: str, params: dict[str, Any] | None = None) -> Any:
    params = params or {}
    name = name.lower().replace("-", "_")

    registry = {
        "dummy": lambda p: DummyClassifier(strategy=p.get("strategy", "most_frequent")),
        "logistic_regression": lambda p: LogisticRegression(max_iter=1000, **p),
        "decision_tree": lambda p: DecisionTreeClassifier(**p),
        "random_forest": lambda p: RandomForestClassifier(random_state=42, **p),
        "hist_gradient_boosting": lambda p: HistGradientBoostingClassifier(random_state=42, **p),
        "mlp_sklearn": lambda p: MLPClassifier(random_state=42, **p),
    }

    if name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            **params,
        )
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(random_state=42, verbose=-1, **params)

    if name not in registry:
        raise ValueError(f"Unknown model: {name}. Available: {list(registry) + ['xgboost', 'lightgbm']}")
    return registry[name](params)


def list_model_names() -> list[str]:
    return [
        "dummy",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
        "lightgbm",
        "mlp_sklearn",
    ]


def get_param_grid(model_name: str) -> dict[str, list]:
    from adult_income_ml.utils import load_model_spaces

    spaces = load_model_spaces()
    key = model_name.lower().replace("-", "_")
    grid = spaces["models"].get(key, {})
    return {k: v if isinstance(v, list) else [v] for k, v in grid.items()}
