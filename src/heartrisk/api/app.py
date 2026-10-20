"""CardioRisk prediction API.

Run locally:
    uvicorn heartrisk.api.app:app --reload --port 8000

Environment variables:
    MODEL_DIR   folder with model.joblib + metadata.json (default: ./models)
    LOG_LEVEL   INFO by default
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from heartrisk import __version__
from heartrisk.api import monitoring
from heartrisk.api.schemas import (
    BatchRequest,
    BatchResponse,
    Health,
    Patient,
    PredictResponse,
)
from heartrisk.models.predict import HeartRiskModel

log = logging.getLogger("cardiorisk.api")
STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    monitoring.configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    model_dir = os.getenv("MODEL_DIR")
    try:
        model = HeartRiskModel.load(model_dir)
        app.state.model = model
        monitoring.MODEL_INFO.labels(model.version, model.metadata.get("model_type", "?")).set(1)
        monitoring.log_event(log, "model loaded", model_version=model.version,
                             model_type=model.metadata.get("model_type"))
    except FileNotFoundError:
        app.state.model = None
        log.exception("model files not found; /ready will report unavailable")
    yield


app = FastAPI(
    title="CardioRisk API",
    description="Predicts the risk of heart disease from routine clinical measurements "
                "(UCI Cleveland data). Returns a label, a probability, a confidence score, "
                "a risk band and the factors that drove the prediction.",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(monitoring.RequestLoggingMiddleware)


def get_model(request: Request) -> HeartRiskModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


def score(model: HeartRiskModel, patients: list[Patient], explain: bool) -> list[dict]:
    start = time.perf_counter()
    try:
        results = model.predict([p.model_dump() for p in patients], explain=explain)
    except Exception:
        monitoring.PREDICTION_ERRORS.inc()
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    monitoring.INFERENCE_LATENCY.observe(time.perf_counter() - start)
    for r in results:
        monitoring.PREDICTIONS.labels(r["label"], r["risk_band"]).inc()
        monitoring.PROBABILITY.observe(r["probability"])
    return results


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=Health, tags=["ops"])
def health(request: Request):
    """Liveness: the process is up (used by the Kubernetes liveness probe)."""
    model = request.app.state.model
    return Health(status="ok", model_loaded=model is not None,
                  model_version=model.version if model else None,
                  uptime_seconds=round(time.time() - STARTED_AT, 1))


@app.get("/ready", response_model=Health, tags=["ops"])
def ready(request: Request, response: Response):
    """Readiness: the model is loaded and can serve traffic."""
    model = request.app.state.model
    if model is None:
        response.status_code = 503
    return Health(status="ok" if model else "unavailable", model_loaded=model is not None,
                  model_version=model.version if model else None,
                  uptime_seconds=round(time.time() - STARTED_AT, 1))


@app.get("/model/info", tags=["model"])
def model_info(request: Request):
    """Version, hyper-parameters and evaluation metrics of the loaded model."""
    return get_model(request).metadata


@app.post("/predict", response_model=PredictResponse, tags=["model"])
def predict(patient: Patient, request: Request,
            explain: bool = Query(True, description="Include the top contributing factors")):
    """Score a single patient."""
    model = get_model(request)
    [result] = score(model, [patient], explain)
    monitoring.log_event(log, "prediction", request_id=request.state.request_id,
                         label=result["label"], probability=result["probability"],
                         risk_band=result["risk_band"], model_version=model.version)
    return PredictResponse(**result, model_version=model.version,
                           request_id=request.state.request_id)


@app.post("/predict/batch", response_model=BatchResponse, tags=["model"])
def predict_batch(batch: BatchRequest, request: Request,
                  explain: bool = Query(False, description="Include the top contributing factors")):
    """Score up to 100 patients in one call."""
    model = get_model(request)
    results = score(model, batch.patients, explain)
    monitoring.log_event(log, "batch prediction", request_id=request.state.request_id,
                         count=len(results),
                         positives=sum(r["prediction"] for r in results),
                         model_version=model.version)
    return BatchResponse(model_version=model.version, request_id=request.state.request_id,
                         count=len(results), predictions=results)


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
