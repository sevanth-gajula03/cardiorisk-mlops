"""API tests against a small model trained on synthetic data (no real model needed)."""

import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from heartrisk.features import build_pipeline


@pytest.fixture(scope="module")
def client(tmp_path_factory, synthetic_patients):
    model_dir = tmp_path_factory.mktemp("model")
    X, y = synthetic_patients
    joblib.dump(build_pipeline(LogisticRegression(max_iter=500)).fit(X, y), model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps({
        "model_version": "test-1", "model_type": "logistic_regression", "decision_threshold": 0.5}))

    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_DIR", str(model_dir))
    from heartrisk.api.app import app
    with TestClient(app) as c:
        yield c
    mp.undo()


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"


def test_health_and_ready(client):
    for path in ("/health", "/ready"):
        body = client.get(path).json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "test-1"


def test_predict_returns_prediction_and_confidence(client, patient):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert 0.5 <= body["confidence"] <= 1
    assert body["risk_band"] in ("low", "moderate", "high")
    assert body["model_version"] == "test-1"
    assert len(body["top_factors"]) == 3
    assert r.headers["x-request-id"] == body["request_id"]


def test_predict_without_explanation(client, patient):
    body = client.post("/predict?explain=false", json=patient).json()
    assert body["top_factors"] == []


def test_request_id_is_propagated(client, patient):
    r = client.post("/predict", json=patient, headers={"x-request-id": "abc123"})
    assert r.json()["request_id"] == "abc123"


@pytest.mark.parametrize("field,value", [
    ("age", 250), ("chol", -5), ("cp", 9), ("thal", 4), ("sex", "male"), ("oldpeak", 12),
])
def test_predict_rejects_invalid_values(client, patient, field, value):
    r = client.post("/predict", json={**patient, field: value})
    assert r.status_code == 422


def test_predict_rejects_missing_and_unknown_fields(client, patient):
    missing = {k: v for k, v in patient.items() if k != "age"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**patient, "bmi": 27}).status_code == 422


def test_batch_predict(client, patient):
    r = client.post("/predict/batch", json={"patients": [patient, {**patient, "age": 45}]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_batch_limits(client, patient):
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422
    assert client.post("/predict/batch", json={"patients": [patient] * 101}).status_code == 422


def test_model_info(client):
    assert client.get("/model/info").json()["model_type"] == "logistic_regression"


def test_metrics_exposes_custom_series(client, patient):
    client.post("/predict", json=patient)
    text = client.get("/metrics").text
    assert "cardiorisk_http_requests_total" in text
    assert "cardiorisk_predictions_total" in text
    assert 'cardiorisk_model_info{model_type="logistic_regression",version="test-1"} 1.0' in text
    assert 'path="/metrics"' not in text


def test_unknown_paths_share_one_metric_label(client):
    client.get("/does-not-exist-1")
    client.get("/does-not-exist-2")
    text = client.get("/metrics").text
    assert 'path="unmatched"' in text
    assert "does-not-exist" not in text
