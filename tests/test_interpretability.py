"""Tests for interpretability helpers."""

from adult_income_ml.features import get_X_y
from adult_income_ml.interpretability import prettify_feature_name, prettify_feature_names
from adult_income_ml.models import get_model
from adult_income_ml.pipelines import build_model_pipeline


def test_prettify_feature_name_numeric():
    assert prettify_feature_name("num__age") == "age"
    assert prettify_feature_name("num__education-num") == "education-num"


def test_prettify_feature_name_categorical():
    assert prettify_feature_name("cat__education_Bachelors") == "education: Bachelors"
    assert prettify_feature_name("cat__sex_Male") == "sex: Male"


def test_prettify_feature_names_list():
    names = ["num__age", "cat__workclass_Private"]
    assert prettify_feature_names(names) == ["age", "workclass: Private"]


def test_get_transformed_feature_names_on_pipeline(sample_raw_df, cfg):
    from adult_income_ml.cleaning import clean_dataframe
    from adult_income_ml.interpretability import get_transformed_feature_names

    clean, _ = clean_dataframe(sample_raw_df, cfg)
    X, y = get_X_y(clean, cfg)
    est = get_model("logistic_regression", {"C": 1.0})
    pipe = build_model_pipeline(est, cfg)
    pipe.fit(X, y)
    names = get_transformed_feature_names(pipe)
    assert len(names) > 0
    assert any("age" in n or "num__age" in n for n in names)
