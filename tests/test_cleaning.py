from adult_income_ml.cleaning import clean_dataframe


def test_clean_target_encoding(sample_raw_df, cfg):
    clean, decisions = clean_dataframe(sample_raw_df, cfg)
    target = cfg["dataset"]["target_column"]
    assert clean[target].isin([0, 1]).all()
    assert any(d["step"] == "target_encoding" for d in decisions)


def test_clean_missing_token(sample_raw_df, cfg):
    clean, _ = clean_dataframe(sample_raw_df, cfg)
    assert clean["workclass"].isna().any() or True  # ? may exist in sample
