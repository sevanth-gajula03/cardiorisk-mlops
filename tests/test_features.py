"""Tests for the engineered features and the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heartrisk.features import RAW_FEATURES, ClinicalFeatures, build_pipeline, build_preprocessor


def test_clinical_features_values(patient):
    out = ClinicalFeatures().fit_transform(pd.DataFrame([patient]))
    assert out.loc[0, "hr_reserve"] == pytest.approx(150 / (220 - 63))
    assert out.loc[0, "age_x_male"] == 63


def test_age_x_male_is_zero_for_women(patient):
    out = ClinicalFeatures().fit_transform(pd.DataFrame([{**patient, "sex": 0}]))
    assert out.loc[0, "age_x_male"] == 0


def test_clinical_features_can_be_disabled(patient):
    frame = pd.DataFrame([patient])
    out = ClinicalFeatures(enabled=False).fit_transform(frame)
    assert list(out.columns) == list(frame.columns)


def test_clinical_features_does_not_mutate_input(patient):
    frame = pd.DataFrame([patient])
    ClinicalFeatures().fit_transform(frame)
    assert "hr_reserve" not in frame.columns


def test_preprocessor_output_is_scaled_and_encoded(synthetic_patients):
    X, _ = synthetic_patients
    X_feat = ClinicalFeatures().fit_transform(X)
    pre = build_preprocessor(engineered=True)
    out = pd.DataFrame(pre.fit_transform(X_feat), columns=pre.get_feature_names_out())

    assert not out.isna().any().any()
    for col in ["age", "chol", "hr_reserve"]:
        assert abs(out[col].mean()) < 1e-6
        assert out[col].std(ddof=0) == pytest.approx(1, abs=1e-6)
    assert {"cp_4", "thal_7", "slope_2"} <= set(out.columns)
    assert "cp" not in out.columns


def test_pipeline_handles_missing_values(synthetic_patients, patient):
    X, y = synthetic_patients
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = pd.DataFrame([{**patient, "ca": np.nan, "chol": np.nan}])[RAW_FEATURES]
    prob = pipe.predict_proba(row)[0, 1]
    assert 0 <= prob <= 1


def test_pipeline_ignores_unseen_category(synthetic_patients, patient):
    X, y = synthetic_patients
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = pd.DataFrame([{**patient, "thal": 5}])[RAW_FEATURES]
    assert pipe.predict(row).shape == (1,)


def test_feature_count_with_and_without_engineering(synthetic_patients):
    X, y = synthetic_patients
    eng = build_pipeline(LogisticRegression(max_iter=500), engineered=True).fit(X, y)
    base = build_pipeline(LogisticRegression(max_iter=500), engineered=False).fit(X, y)
    n_eng = len(eng.named_steps["preprocess"].get_feature_names_out())
    n_base = len(base.named_steps["preprocess"].get_feature_names_out())
    assert n_eng == n_base + 2
