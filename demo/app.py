#!/usr/bin/env python
"""Gradio demo for Adult Income prediction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import joblib
import pandas as pd

from adult_income_ml.features import get_feature_columns
from adult_income_ml.interpretability import (
    get_transformed_feature_names,
    prettify_feature_names,
)
from adult_income_ml.utils import get_paths, load_config

DISCLAIMER = (
    "**Disclaimer:** This tool is for research and educational purposes only. "
    "Predictions are based on historical U.S. census data and must not be used for "
    "employment, credit, lending, or other high-stakes decisions. The model may "
    "reflect biases present in the training data."
)

MODEL_MISSING_MSG = "**Run make tune first** — `results/models/final_model.joblib` was not found."

# UCI Adult dataset fallbacks when processed data is unavailable.
UCI_CATEGORICAL_CHOICES: dict[str, list[str]] = {
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "marital-status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ],
    "race": [
        "White",
        "Black",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
    ],
    "sex": ["Male", "Female"],
    "native-country": [
        "United-States",
        "Mexico",
        "Philippines",
        "Germany",
        "Canada",
        "El-Salvador",
        "India",
        "Cuba",
        "England",
        "China",
        "Jamaica",
        "South",
        "Italy",
        "Dominican-Republic",
        "Japan",
        "Guatemala",
        "Vietnam",
        "Columbia",
        "Poland",
        "Haiti",
        "Portugal",
        "Iran",
        "Taiwan",
        "Nicaragua",
        "Peru",
        "France",
        "Greece",
        "Ecuador",
        "Ireland",
        "Hong",
        "Cambodia",
        "Trinadad&Tobago",
        "Laos",
        "Thailand",
        "Yugoslavia",
        "Outlying-US(Guam-USVI-etc)",
        "Hungary",
        "Honduras",
        "Scotland",
        "Holand-Netherlands",
    ],
}

NUMERIC_DEFAULTS: dict[str, float] = {
    "age": 39.0,
    "fnlwgt": 200_000.0,
    "education-num": 13.0,
    "capital-gain": 0.0,
    "capital-loss": 0.0,
    "hours-per-week": 40.0,
}

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (17, 90),
    "fnlwgt": (10_000, 1_500_000),
    "education-num": (1, 16),
    "capital-gain": (0, 99_999),
    "capital-loss": (0, 4_356),
    "hours-per-week": (1, 99),
}

CAT_DEFAULTS: dict[str, str] = {
    "workclass": "Private",
    "education": "Bachelors",
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "native-country": "United-States",
}


def _load_model() -> object | None:
    cfg = load_config()
    paths = get_paths(cfg)
    model_path = paths["models"] / "final_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def _load_background_df(cfg: dict) -> pd.DataFrame | None:
    paths = get_paths(cfg)
    clean_path = paths["processed"] / cfg["dataset"]["clean_filename"]
    if not clean_path.exists():
        return None
    feat = get_feature_columns(cfg)
    cols = feat["numeric"] + feat["categorical"]
    df = pd.read_csv(clean_path, usecols=lambda c: c in cols)
    return df


def _categorical_choices(cfg: dict) -> dict[str, list[str]]:
    background = _load_background_df(cfg)
    feat = get_feature_columns(cfg)
    choices: dict[str, list[str]] = {}
    for col in feat["categorical"]:
        if background is not None and col in background.columns:
            values = sorted(background[col].dropna().astype(str).unique().tolist())
            choices[col] = values if values else UCI_CATEGORICAL_CHOICES.get(col, [])
        else:
            choices[col] = UCI_CATEGORICAL_CHOICES.get(col, [])
    return choices


def _reference_values(cfg: dict, background: pd.DataFrame | None) -> dict[str, object]:
    feat = get_feature_columns(cfg)
    ref: dict[str, object] = {}
    for col in feat["numeric"]:
        if background is not None and col in background.columns:
            ref[col] = float(background[col].median())
        else:
            ref[col] = NUMERIC_DEFAULTS[col]
    for col in feat["categorical"]:
        if background is not None and col in background.columns:
            ref[col] = background[col].mode(dropna=True).iloc[0]
        else:
            ref[col] = CAT_DEFAULTS[col]
    return ref


def _build_input_row(cfg: dict, *values) -> pd.DataFrame:
    feat = get_feature_columns(cfg)
    columns = feat["numeric"] + feat["categorical"]
    return pd.DataFrame([dict(zip(columns, values))], columns=columns)


def _permutation_single_row(
    model,
    row: pd.DataFrame,
    reference: dict[str, object],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    baseline = float(model.predict_proba(row)[0, 1])
    impacts: list[tuple[str, float]] = []
    for col in row.columns:
        perturbed = row.copy()
        perturbed[col] = reference[col]
        proba = float(model.predict_proba(perturbed)[0, 1])
        impacts.append((col, baseline - proba))
    impacts.sort(key=lambda item: abs(item[1]), reverse=True)
    return impacts[:top_k]


def _shap_single_row(model, row: pd.DataFrame, top_k: int = 5) -> list[tuple[str, float]] | None:
    if not hasattr(model, "named_steps"):
        return None
    try:
        import shap
    except ImportError:
        return None

    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    x_t = pre.transform(row)
    raw_names = get_transformed_feature_names(model)
    pretty_names = prettify_feature_names(raw_names) if raw_names else None

    try:
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(x_t)
        elif hasattr(clf, "coef_"):
            explainer = shap.LinearExplainer(clf, x_t)
            shap_values = explainer.shap_values(x_t)
        else:
            return None
    except Exception:
        return None

    if isinstance(shap_values, list):
        shap_row = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        shap_row = shap_values[0] if getattr(shap_values, "ndim", 1) > 1 else shap_values

    names = (
        pretty_names
        if pretty_names and len(pretty_names) == len(shap_row)
        else [f"feature_{i}" for i in range(len(shap_row))]
    )
    ranked = sorted(zip(names, shap_row), key=lambda item: abs(float(item[1])), reverse=True)
    return [(name, float(value)) for name, value in ranked[:top_k]]


def _format_top_features(features: list[tuple[str, float]] | None) -> str:
    if not features:
        return "_Feature attribution unavailable for this model._"
    lines = ["| Feature | Impact |", "| --- | ---: |"]
    for name, value in features:
        lines.append(f"| {name} | {value:+.4f} |")
    return "\n".join(lines)


def predict(
    model,
    cfg: dict,
    background: pd.DataFrame | None,
    reference: dict[str, object],
    pos_label: str,
    neg_label: str,
    *values,
) -> tuple[str, str, str]:
    if model is None:
        return MODEL_MISSING_MSG, "", DISCLAIMER

    row = _build_input_row(cfg, *values)
    pred_idx = int(model.predict(row)[0])
    classes = list(model.classes_)
    label = classes[pred_idx]
    proba = model.predict_proba(row)[0]
    pos_idx = classes.index(">50K") if ">50K" in classes else 1
    p_high = float(proba[pos_idx])

    prediction_md = (
        f"**Predicted income:** `{label}` "
        f"({'high income' if label == pos_label else 'low income'})"
    )
    probability_md = (
        f"**P({pos_label}):** {p_high:.1%}  \n"
        f"**P({neg_label}):** {1.0 - p_high:.1%}"
    )

    top_features = _shap_single_row(model, row, top_k=5)
    if top_features is None:
        top_features = _permutation_single_row(model, row, reference, top_k=5)

    explain_md = "### Top contributing features\n\n" + _format_top_features(top_features)
    return prediction_md, probability_md + "\n\n" + explain_md, DISCLAIMER


def build_demo() -> gr.Blocks:
    cfg = load_config()
    feat = get_feature_columns(cfg)
    model = _load_model()
    background = _load_background_df(cfg)
    reference = _reference_values(cfg, background)
    cat_choices = _categorical_choices(cfg)
    pos_label = cfg["dataset"]["positive_label"]
    neg_label = cfg["dataset"]["negative_label"]

    with gr.Blocks(title="Adult Income Predictor") as demo:
        gr.Markdown("# Adult Income Prediction Demo")
        gr.Markdown(
            "Enter census-style features to predict whether annual income is "
            f"`{neg_label}` or `{pos_label}`."
        )

        if model is None:
            gr.Markdown(MODEL_MISSING_MSG)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Numeric features")
                numeric_inputs = []
                for col in feat["numeric"]:
                    lo, hi = NUMERIC_BOUNDS[col]
                    default = (
                        float(background[col].median())
                        if background is not None and col in background.columns
                        else NUMERIC_DEFAULTS[col]
                    )
                    numeric_inputs.append(
                        gr.Number(
                            label=col,
                            value=default,
                            minimum=lo,
                            maximum=hi,
                        )
                    )

            with gr.Column():
                gr.Markdown("### Categorical features")
                categorical_inputs = []
                for col in feat["categorical"]:
                    options = cat_choices.get(col, UCI_CATEGORICAL_CHOICES.get(col, []))
                    default = CAT_DEFAULTS.get(col, options[0] if options else "")
                    if default not in options and options:
                        options = [default, *options]
                    categorical_inputs.append(
                        gr.Dropdown(
                            label=col,
                            choices=options,
                            value=default,
                        )
                    )

        predict_btn = gr.Button("Predict", variant="primary", interactive=model is not None)
        prediction_out = gr.Markdown(label="Prediction")
        details_out = gr.Markdown(label="Details")
        disclaimer_out = gr.Markdown(DISCLAIMER)

        all_inputs = numeric_inputs + categorical_inputs
        predict_btn.click(
            fn=lambda *vals: predict(
                model, cfg, background, reference, pos_label, neg_label, *vals
            ),
            inputs=all_inputs,
            outputs=[prediction_out, details_out, disclaimer_out],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
