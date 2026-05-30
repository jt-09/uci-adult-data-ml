from adult_income_ml.cleaning import clean_dataframe
from adult_income_ml.features import build_feature_dictionary, get_X_y, get_feature_columns


def test_feature_columns(cfg):
    feat = get_feature_columns(cfg)
    assert "age" in feat["numeric"]
    assert "sex" in feat["sensitive"]


def test_get_X_y_shape(sample_raw_df, cfg):
    clean, _ = clean_dataframe(sample_raw_df, cfg)
    X, y = get_X_y(clean, cfg)
    assert len(X) == len(y)
    assert "income" not in X.columns


def test_drop_sensitive(sample_raw_df, cfg):
    clean, _ = clean_dataframe(sample_raw_df, cfg)
    X, _ = get_X_y(clean, cfg, drop_sensitive=True)
    assert "sex" not in X.columns
    assert "race" not in X.columns


def test_feature_dictionary(cfg):
    fd = build_feature_dictionary(cfg)
    assert "feature" in fd.columns
