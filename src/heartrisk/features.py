"""Feature engineering and preprocessing, packaged as a single sklearn pipeline.

The pipeline takes the cleaned data frame (integer codes, one row per patient)
and handles everything else - derived features, imputation, scaling and
encoding - so the exact same object is used in training, in the API and in
batch inference.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from heartrisk.config import load_config

RAW_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds domain features derived from the raw measurements.

    hr_reserve  - achieved max heart rate as a fraction of the age-predicted
                  maximum (220 - age). Healthy hearts get closer to 1.0.
    age_x_male  - age for men, 0 for women. Lets linear models learn that risk
                  rises with age much faster for men than for women.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.enabled:
            X["hr_reserve"] = X["thalach"] / (220 - X["age"]).clip(lower=1)
            X["age_x_male"] = X["age"] * X["sex"]
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.enabled:
            names += ["hr_reserve", "age_x_male"]
        return np.asarray(names, dtype=object)


def category_name(feature: str, category) -> str:
    """One-hot column names like cp_4 instead of cp_4.0 (inputs arrive as floats)."""
    if isinstance(category, float) and category.is_integer():
        category = int(category)
    return f"{feature}_{category}"


def build_preprocessor(engineered: bool = True) -> ColumnTransformer:
    cfg = load_config()
    cols = cfg["columns"]
    skewed = cfg["features"]["skewed"]

    scaled = [c for c in cols["numeric"] if c not in skewed] + cols["ordinal"]
    if engineered:
        scaled += cfg["features"]["engineered"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaled", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), scaled),
            ("skewed", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("power", PowerTransformer(method="yeo-johnson")),
            ]), skewed),
            ("binary", SimpleImputer(strategy="most_frequent"), cols["binary"]),
            ("nominal", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                         feature_name_combiner=category_name)),
            ]), cols["nominal"]),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_pipeline(estimator, engineered: bool = True) -> Pipeline:
    """Full model pipeline: raw clinical columns in, prediction out."""
    pipe = Pipeline([
        ("features", ClinicalFeatures(enabled=engineered)),
        ("preprocess", build_preprocessor(engineered)),
        ("model", estimator),
    ])
    pipe.set_output(transform="pandas")
    return pipe
