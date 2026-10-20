"""Tests for model training helpers, evaluation and the inference wrapper."""

import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from heartrisk.features import build_pipeline
from heartrisk.models import evaluate
from heartrisk.models.predict import HeartRiskModel, risk_band
from heartrisk.models.train import make_estimator, param_grid


@pytest.fixture
def fitted_pipeline(synthetic_patients):
    X, y = synthetic_patients
    return build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)


@pytest.fixture
def saved_model_dir(tmp_path, fitted_pipeline):
    joblib.dump(fitted_pipeline, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({
        "model_version": "test", "model_type": "logistic_regression", "decision_threshold": 0.5}))
    return tmp_path


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_make_estimator_known_models(name):
    est = make_estimator(name, seed=1)
    assert hasattr(est, "fit") and hasattr(est, "predict_proba")


def test_make_estimator_rejects_unknown():
    with pytest.raises(ValueError):
        make_estimator("svm", seed=1)


def test_param_grid_prefixes_model_step():
    assert param_grid({"C": [1, 2]}) == {"model__C": [1, 2]}


def test_compute_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    metrics = evaluate.compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"}


def test_compute_metrics_respects_threshold():
    y = np.array([0, 1])
    probs = np.array([0.3, 0.6])
    assert evaluate.compute_metrics(y, probs, threshold=0.5)["recall"] == 1.0
    assert evaluate.compute_metrics(y, probs, threshold=0.7)["recall"] == 0.0


def test_feature_importance_covers_all_features(fitted_pipeline):
    imp = evaluate.feature_importance(fitted_pipeline)
    names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    assert set(imp.index) == set(names)


def test_model_learns_signal(synthetic_patients, fitted_pipeline):
    X, y = synthetic_patients
    probs = fitted_pipeline.predict_proba(X)[:, 1]
    assert evaluate.compute_metrics(y, probs)["roc_auc"] > 0.8


@pytest.mark.parametrize("prob,band", [(0.05, "low"), (0.29, "low"), (0.3, "moderate"),
                                       (0.59, "moderate"), (0.6, "high"), (1.0, "high")])
def test_risk_band(prob, band):
    assert risk_band(prob) == band


def test_wrapper_predict_output_shape(saved_model_dir, patient):
    model = HeartRiskModel.load(saved_model_dir)
    [result] = model.predict([patient])
    assert set(result) == {"prediction", "label", "probability", "confidence", "risk_band"}
    assert result["prediction"] in (0, 1)
    assert 0.5 <= result["confidence"] <= 1.0
    expected = result["probability"] if result["prediction"] else 1 - result["probability"]
    assert result["confidence"] == pytest.approx(expected, abs=1e-4)


def test_wrapper_rejects_missing_fields(saved_model_dir, patient):
    model = HeartRiskModel.load(saved_model_dir)
    incomplete = {k: v for k, v in patient.items() if k != "thalach"}
    with pytest.raises(ValueError, match="thalach"):
        model.predict([incomplete])


def test_wrapper_batch_prediction(saved_model_dir, patient):
    model = HeartRiskModel.load(saved_model_dir)
    assert len(model.predict([patient, {**patient, "age": 40}])) == 2
