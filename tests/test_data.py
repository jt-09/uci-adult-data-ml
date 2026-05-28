from adult_income_ml.data import ADULT_COLUMNS


def test_adult_columns_count():
    assert len(ADULT_COLUMNS) == 15


def test_sample_raw_schema(sample_raw_df):
    assert list(sample_raw_df.columns) == ADULT_COLUMNS
    assert len(sample_raw_df) >= 10
