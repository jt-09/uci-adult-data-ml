from sklearn.model_selection import train_test_split

from adult_income_ml.cleaning import clean_dataframe
from adult_income_ml.features import get_X_y
from adult_income_ml.models import get_model
from adult_income_ml.pipelines import build_model_pipeline


def test_pipeline_fit_transform(sample_raw_df, cfg):
    clean, _ = clean_dataframe(sample_raw_df, cfg)
    X, y = get_X_y(clean, cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    pipe = build_model_pipeline(get_model("logistic_regression", {}), cfg)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    assert len(pred) == len(y_test)


def test_pipeline_drop_sensitive(sample_raw_df, cfg):
    clean, _ = clean_dataframe(sample_raw_df, cfg)
    X, y = get_X_y(clean, cfg, drop_sensitive=True)
    pipe = build_model_pipeline(get_model("dummy", {}), cfg, drop_sensitive=True)
    pipe.fit(X, y)
    assert pipe.predict(X).shape[0] == len(y)
