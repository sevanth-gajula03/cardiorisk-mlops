# syntax=docker/dockerfile:1
# CardioRisk API image
#   docker build -t cardiorisk-api:latest .
#   docker run --rm -p 8000:8000 cardiorisk-api:latest

# ---- build stage: compile wheels so the runtime image stays small ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements-serve.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-serve.txt

COPY pyproject.toml .
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ---- runtime stage ----
FROM python:3.11-slim

ARG MODEL_VERSION=unknown
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="cardiorisk-api" \
      org.opencontainers.image.description="Heart disease risk prediction API" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      cardiorisk.model.version="${MODEL_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HEARTRISK_HOME=/app \
    MODEL_DIR=/app/models \
    LOG_LEVEL=INFO \
    PORT=8000

RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home /app app

WORKDIR /app
# Mount the wheels instead of copying them so they never end up in an image layer.
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*

COPY config ./config
COPY models/model.joblib models/metadata.json ./models/

# Numeric UID so Kubernetes can verify runAsNonRoot.
USER 10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)" || exit 1

# One worker per container: Prometheus counters are per-process, so scaling is done
# with Kubernetes replicas (each pod scraped separately) rather than uvicorn workers.
CMD ["sh", "-c", "uvicorn heartrisk.api.app:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
