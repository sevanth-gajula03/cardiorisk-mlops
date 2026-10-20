"""Prometheus metrics and structured request logging for the API."""

import json
import logging
import sys
import time
import uuid

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUESTS = Counter(
    "cardiorisk_http_requests_total", "HTTP requests handled",
    ["method", "path", "status"])
LATENCY = Histogram(
    "cardiorisk_http_request_duration_seconds", "End-to-end request latency",
    ["method", "path"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
IN_FLIGHT = Gauge("cardiorisk_http_requests_in_flight", "Requests currently being served")

PREDICTIONS = Counter(
    "cardiorisk_predictions_total", "Predictions returned, by outcome",
    ["label", "risk_band"])
PROBABILITY = Histogram(
    "cardiorisk_prediction_probability", "Distribution of predicted disease probability",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
INFERENCE_LATENCY = Histogram(
    "cardiorisk_inference_duration_seconds", "Time spent inside the model only",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1))
PREDICTION_ERRORS = Counter(
    "cardiorisk_prediction_errors_total", "Requests that failed inside the model")
MODEL_INFO = Gauge(
    "cardiorisk_model_info", "Currently loaded model (value is always 1)",
    ["version", "model_type"])

# Keep high-cardinality or noisy paths out of the request metrics.
UNTRACKED_PATHS = {"/metrics"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, easy to grep locally and to ship to a log stack."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn's own access log would duplicate ours.
    logging.getLogger("uvicorn.access").disabled = True


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Adds a request id, records Prometheus metrics and writes one log line per request."""

    def __init__(self, app, logger_name: str = "cardiorisk.access"):
        super().__init__(app)
        self.log = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        IN_FLIGHT.inc()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            IN_FLIGHT.dec()
            elapsed = time.perf_counter() - start
            # Use the matched route template; unknown URLs share one label so random
            # 404 traffic cannot create unbounded metric series.
            route = request.scope.get("route")
            path = route.path if route is not None else "unmatched"
            if path not in UNTRACKED_PATHS:
                REQUESTS.labels(request.method, path, str(status)).inc()
                LATENCY.labels(request.method, path).observe(elapsed)
                log_event(self.log, "request", request_id=request_id, method=request.method,
                          path=path, status=status, latency_ms=round(elapsed * 1000, 2),
                          client=request.client.host if request.client else None)
