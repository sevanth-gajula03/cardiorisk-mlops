# CardioRisk - Heart Disease Risk Prediction (MLOps)

End-to-end MLOps project for **AIMLCZG523 - Machine Learning Operations, Assignment 1**.
A classifier predicts the presence of heart disease from routine clinical measurements
and is packaged, tested, containerised and deployed with production-style tooling.

## Project layout

```
.
├── config/params.yaml          # single source of truth for paths, columns, ranges
├── data/
│   ├── raw/                    # downloaded UCI file + MANIFEST.json (sha256)
│   └── processed/              # heart_clean.csv, heart_encoded.csv
├── models/                     # packaged champion model (joblib + MLflow format)
├── notebooks/01_eda.ipynb      # executed EDA notebook
├── reports/
│   ├── data_quality.json       # what cleaning changed
│   ├── model_comparison.md     # auto-generated results table
│   └── figures/                # EDA + model plots used in the report
├── samples/patients.json       # example inference input
├── scripts/                    # deploy, smoke test, load test, notebook builder
├── src/heartrisk/
│   ├── config.py
│   ├── data/download.py        # dataset acquisition
│   ├── data/clean.py           # cleaning + encoding
│   ├── eda/plots.py            # all EDA figures
│   ├── api/                    # FastAPI app, request/response schemas, monitoring
│   ├── features.py             # engineered features + preprocessing pipeline
│   └── models/
│       ├── train.py            # tuning, MLflow tracking, packaging
│       ├── evaluate.py         # metrics and diagnostic plots
│       └── predict.py          # inference wrapper + CLI
├── tests/                      # pytest unit tests
├── screenshots/                # evidence for the report
├── .github/workflows/ci.yml    # lint -> test -> train -> container pipeline
├── k8s/
│   ├── base/                   # namespace, config, deployment, service, ingress, HPA, PDB
│   ├── overlays/minikube/      # pins the image tag to the model version
│   └── monitoring/             # Prometheus (scrape config, alert rules) + Grafana (dashboard)
├── Dockerfile                  # multi-stage, non-root API image
├── Makefile
├── environment.yml             # conda alternative
├── requirements.txt            # full dev / training environment
└── requirements-serve.txt      # slim runtime deps for the container
```

## Setup

Python 3.10+ is required (developed on 3.11).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Part 1 - Data acquisition & EDA

### Getting the data

```bash
python -m heartrisk.data.download          # add --force to re-download
```

The script downloads the official archive from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease),
extracts `processed.cleveland.data` (303 patients) and writes `data/raw/MANIFEST.json`
with the source URL, timestamp and SHA-256 checksum. If the archive endpoint fails it
falls back to the direct file link. TLS is verified against the OS certificate store,
so it also works behind corporate proxies.

**Manual alternative:** download `processed.cleveland.data` from the UCI page above and
place it at `data/raw/processed.cleveland.data`.

### Cleaning

```bash
python -m heartrisk.data.clean
```

| Step | Detail |
|---|---|
| Missing values | `?` parsed as NaN; `ca` (4 rows) and `thal` (2 rows) imputed with the mode |
| Duplicates | exact duplicates dropped (none in the Cleveland file) |
| Range checks | readings outside clinically plausible limits in `config/params.yaml` are treated as missing |
| Target | `num` 0-4 collapsed to binary `target` (0 = no disease, 1 = disease) |
| Encoding | `heart_encoded.csv` one-hot encodes `cp`, `restecg`, `slope`, `thal` with readable names |

Scaling is intentionally left to the model pipeline so it is fitted on training folds only.
Every change is recorded in `reports/data_quality.json`.

### EDA

```bash
python -m heartrisk.eda.plots              # writes reports/figures/*.png
make notebook                              # rebuilds and executes notebooks/01_eda.ipynb
```

| Figure | Purpose |
|---|---|
| `01_class_balance` | 164 healthy vs 139 diseased (54/46) |
| `02_missing_values` | missingness in the raw file |
| `03_numeric_histograms` | continuous features split by target |
| `04_numeric_boxplots` | spread and outliers |
| `05_correlation_heatmap` | pairwise correlations |
| `06_target_correlation` | ranked correlation with the target |
| `07_categorical_disease_rate` | disease rate per category with group sizes |
| `08_age_band_risk` | disease rate by age band and sex |
| `09_hr_vs_age` | max heart rate vs the `220 - age` reference |

Key takeaways: `thal`, `ca`, `exang`, `oldpeak`, `cp` and `thalach` carry the most signal;
`chol` and `fbs` are weak; `oldpeak` is right-skewed; risk rises with age and is much higher
for men. These motivate the engineered features in Part 2 (heart-rate reserve, age x sex).

## Part 2 - Feature engineering & model development

### Features

Everything lives in one sklearn `Pipeline` ([features.py](src/heartrisk/features.py)), so the
exact same transformations run in training, the API and batch scoring:

| Step | Columns | Why |
|---|---|---|
| `ClinicalFeatures` | adds `hr_reserve = thalach / (220 - age)` and `age_x_male = age * sex` | EDA showed peak heart rate depends on age, and risk rises with age much faster for men |
| median impute + `StandardScaler` | age, trestbps, chol, thalach, ca, engineered | puts features on one scale for linear models |
| median impute + Yeo-Johnson | oldpeak | heavily right-skewed |
| mode impute | sex, fbs, exang | already 0/1 |
| mode impute + one-hot | cp, restecg, slope, thal | codes are not ordered; unseen codes are ignored safely |

### Models and tuning

Three model families are tuned with `GridSearchCV` (5-fold stratified CV, scoring ROC-AUC,
plus accuracy, precision, recall, F1). Search spaces are in `config/params.yaml`:

- **Logistic regression** - `C`, L1/L2 penalty, class weighting (24 candidates)
- **Random forest** - trees, depth, min leaf size, max features (36 candidates)
- **Gradient boosting** - trees, learning rate, depth, subsample (16 candidates)

Each model is trained **twice - with and without the engineered features** - to measure
whether they actually help (feature ablation).

**Protocol:** stratified 80/20 split (seed 42). Tuning and model selection use only the
training split (CV). The 20% hold-out is reported but never used to choose anything.
The chosen configuration is then refit on all 303 rows for the shipped model.

### Results

| Run | CV ROC-AUC | Test ROC-AUC | Test recall | Train-CV gap |
|---|---|---|---|---|
| **logreg-eng** (champion) | **0.915 +/- 0.013** | 0.948 | 0.929 | 0.010 |
| logreg-base | 0.908 +/- 0.014 | 0.953 | 0.929 | 0.025 |
| rf-eng | 0.903 +/- 0.021 | 0.945 | 0.893 | 0.067 |
| rf-base | 0.900 +/- 0.026 | 0.950 | 0.929 | 0.064 |
| gbm-eng | 0.894 +/- 0.024 | 0.956 | 0.929 | 0.073 |
| gbm-base | 0.883 +/- 0.028 | 0.960 | 0.929 | 0.078 |

Full table: [reports/model_comparison.md](reports/model_comparison.md).

**Why logistic regression (C = 0.1, L2) wins:**
- Highest and most stable CV ROC-AUC (smallest std).
- Smallest train-vs-CV gap (0.01) - the tree models overfit 303 rows noticeably (~0.07).
- Probabilities track observed rates reasonably well (slightly cautious at the low end, see the
  calibration plot), and the coefficients are readable - both matter for a clinical risk score.
- Gradient boosting scores higher on the 61-row test set, but its CV score is the lowest; with
  so few test rows that difference is within noise, so I trust CV.

**Did feature engineering help?** Yes - CV ROC-AUC improved for all three families
(+0.007 logreg, +0.003 RF, +0.011 GBM).

## Part 3 - Experiment tracking (MLflow)

```bash
make train          # or: python -m heartrisk.models.train
make mlflow-ui      # http://localhost:5000  -> experiment "cardiorisk-heart-disease"
```

Each training session is a parent run with one nested run per model / feature set.

| Logged | Where |
|---|---|
| Parameters | model type, feature set, best hyper-parameters, grid size, CV folds, seed, data SHA-256 |
| Metrics | CV mean and std for accuracy/precision/recall/F1/ROC-AUC, train ROC-AUC, overfit gap, test metrics |
| Plots | confusion matrix, ROC + PR curves, calibration curve, feature importance (per run); comparison + ROC overlay (parent) |
| Artifacts | `cv_results.csv`, `feature_importance.csv`, `search_space.json`, `params.yaml`, model comparison |
| Dataset | `heart_clean` registered as the training input |
| Models | every tuned pipeline with signature + input example; champion registered as `cardiorisk-classifier` with alias `@champion` |

Set `MLFLOW_TRACKING_URI` to log to a remote server instead of the local `mlflow.db`.

## Part 4 - Packaging & reproducibility

`models/` holds the deployable champion:

| File | Contents |
|---|---|
| `model.joblib` | full sklearn pipeline (features + preprocessing + model) |
| `mlflow_model/` | the same pipeline in MLflow format (`MLmodel`, `conda.yaml`, `requirements.txt`, input example) |
| `metadata.json` | version, hyper-parameters, metrics, feature list, training-data SHA-256, MLflow run ids |

Three equivalent ways to load it:

```python
from heartrisk.models.predict import HeartRiskModel
HeartRiskModel.load().predict([{...patient...}])

import mlflow
mlflow.pyfunc.load_model("models/mlflow_model")
mlflow.sklearn.load_model("models:/cardiorisk-classifier@champion")
```

Command line:

```bash
make predict        # python -m heartrisk.models.predict --input samples/patients.json
```

Each prediction returns `prediction`, `label`, `probability`, `confidence` and a `risk_band`
(low < 0.3 <= moderate < 0.6 <= high).

**Reproducibility:** pinned `requirements.txt` (or `environment.yml`), one random seed in
config, a deterministic split, the data checksum recorded in the manifest and model metadata,
and preprocessing stored inside the model so nothing has to be re-implemented at serving time.

## Part 5 - CI/CD pipeline (GitHub Actions)

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to `main`, on pull
requests and on demand:

```
lint ──► test ──► train ──► docker
manifests (in parallel)
```

| Job | What it does | Artifacts / output |
|---|---|---|
| **Lint** | `ruff check` (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify), annotations inline on the PR | - |
| **Validate Kubernetes manifests** | renders `k8s/overlays/minikube` and `k8s/monitoring` with kustomize and validates them with kubeconform (strict) | - |
| **Unit tests** | 60 pytest tests with coverage | `test-reports` (JUnit XML, coverage XML + HTML); coverage % in the job summary |
| **Train and evaluate** | downloads the data, cleans it, regenerates EDA figures, runs the full tuning session with MLflow, applies the **model quality gate**, scores the sample patients | `model`, `reports`, `mlflow-runs` (`mlflow.db` + `mlruns/`); comparison table and gate result in the job summary |
| **Build and smoke-test container** | builds the image **with the model trained in this run**, starts it and runs `scripts/smoke_test.sh`; on `main` pushes `ghcr.io/<owner>/cardiorisk-api:<model-version>` and `:latest` | `container-logs`; image size in the summary |

**Failing fast:** each job depends on the previous one, every step runs with `set -e`
semantics, and the run fails if lint finds an issue, a test fails, the model misses the quality
gate (`config/params.yaml -> training.quality_gate`: CV ROC-AUC >= 0.85, test ROC-AUC >= 0.85,
test recall >= 0.80) or any smoke check fails. Logs for each step are in the Actions UI and the
artifacts above.

Run the same checks locally:

```bash
ruff check .
pytest --cov
python scripts/check_model_quality.py
```

## Part 6 - Prediction API & container

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | score one patient; `?explain=false` skips the factor breakdown |
| POST | `/predict/batch` | score 1-100 patients |
| GET | `/health` | liveness - process is up |
| GET | `/ready` | readiness - model loaded (503 otherwise) |
| GET | `/model/info` | version, hyper-parameters, metrics of the loaded model |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | interactive Swagger UI |

Example:

```bash
curl -X POST http://localhost:8080/predict -H 'content-type: application/json' -d '{
  "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
  "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3}'
```

```json
{
  "prediction": 1, "label": "heart_disease", "probability": 0.9874, "confidence": 0.9874,
  "risk_band": "high",
  "top_factors": [
    {"feature": "ca", "effect": "raises_risk", "weight": 1.86},
    {"feature": "cp_4", "effect": "raises_risk", "weight": 0.619},
    {"feature": "thal_3", "effect": "lowers_risk", "weight": -0.514}
  ],
  "model_version": "2026.10.17.1617", "request_id": "b560eecfba3c"
}
```

Design notes:
- **Validation** - every field has clinical limits and allowed codes (Pydantic); unknown fields are
  rejected, so bad input returns a clear 422 instead of a silent wrong prediction.
- **Explanations** - `top_factors` are the three largest coefficient x feature contributions to
  this patient's log-odds, relative to an average patient.
- **Traceability** - every response carries `model_version` and a `request_id` (also returned as
  the `X-Request-ID` header and accepted from callers), which links a response to its log line.

### Run it

```bash
make serve                      # local dev server with reload on :8080
make docker-build               # image tagged cardiorisk-api:latest
make docker-run                 # container on http://localhost:8080
make smoke                      # 6 end-to-end checks against the running API
```

Image details: multi-stage build (wheels compiled in a builder stage and bind-mounted, so they
never land in a layer), slim runtime requirements, runs as a non-root `app` user, Docker
`HEALTHCHECK` on `/health`, model version and git SHA stored as image labels. One uvicorn worker
per container - Prometheus counters are per-process, so scaling is done with Kubernetes replicas.

## Part 7 - Kubernetes deployment (Minikube)

### Cluster setup (one time)

```bash
brew install colima docker docker-buildx kubectl minikube helm
colima start --cpu 4 --memory 8 --disk 40
minikube start --driver=docker --cpus=3 --memory=5g
minikube addons enable ingress
minikube addons enable metrics-server
```

> **Behind a TLS-inspecting corporate proxy?** Copy the company root/issuing CA certificates
> (PEM) into `~/.minikube/certs/` and run `minikube start --embed-certs`. For the Colima Docker
> engine, add them to `/usr/local/share/ca-certificates/` inside `colima ssh`, run
> `sudo update-ca-certificates` and restart Docker. Images that still fail to pull inside the
> cluster can be pulled on the host and pushed in with `minikube image load <image>`.

### Deploy

```bash
make deploy        # scripts/deploy_minikube.sh
```

The script builds the image tagged with the **model version** (e.g.
`cardiorisk-api:2026.10.17.1617`), loads it into Minikube (no registry needed), pins that tag in the
kustomize overlay, applies the manifests, waits for the rollout, records a change-cause and runs
the smoke test through the Ingress. Retraining therefore always produces a new tag and a
zero-downtime rolling update; `kubectl rollout undo` goes back to the previous model.

### What gets deployed (`k8s/base`)

| Resource | Notes |
|---|---|
| Namespace `cardiorisk` | everything is isolated in one namespace |
| ConfigMap | `LOG_LEVEL`, `MODEL_DIR` |
| Deployment | 2 replicas, rolling update with `maxUnavailable: 0`, startup / liveness (`/health`) / readiness (`/ready`) probes, CPU + memory requests and limits, non-root UID 10001, read-only root filesystem, all capabilities dropped, Prometheus scrape annotations |
| Service (LoadBalancer) | port 8090; gets an external IP from `minikube tunnel` or a cloud LB |
| Ingress (nginx) | host `cardiorisk.local`, request-size and rate limits |
| HorizontalPodAutoscaler | 2-5 pods at 70% CPU |
| PodDisruptionBudget | at least one pod always available |

### Access

```bash
# LoadBalancer
minikube tunnel                                   # keep running in a separate terminal
curl http://127.0.0.1:8090/health

# Ingress
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8081:80
curl -H 'Host: cardiorisk.local' http://localhost:8081/health
# or add "127.0.0.1 cardiorisk.local" to /etc/hosts and open http://cardiorisk.local:8081/docs

make k8s-status    # deployments, pods, services, ingress, HPA, rollout history
make load          # 20 req/s for 2 minutes, for dashboards / autoscaling demos
```

## Part 8 - Monitoring & logging

### Request logging

Every request produces one JSON log line (`cardiorisk.access`) with request id, method, route,
status, latency and client; every prediction adds a `cardiorisk.api` line with label, probability,
risk band and model version. The request id is taken from the caller / ingress `X-Request-ID`
header when present, so one id can be followed from nginx to the pod log to the response.

```bash
kubectl -n cardiorisk logs -l app.kubernetes.io/name=cardiorisk-api --tail=20 -f
```

### Metrics (`/metrics`)

| Metric | Type | Labels |
|---|---|---|
| `cardiorisk_http_requests_total` | counter | method, path (route template), status |
| `cardiorisk_http_request_duration_seconds` | histogram | method, path |
| `cardiorisk_http_requests_in_flight` | gauge | - |
| `cardiorisk_predictions_total` | counter | label, risk_band |
| `cardiorisk_prediction_probability` | histogram | - |
| `cardiorisk_inference_duration_seconds` | histogram | - (model time only) |
| `cardiorisk_prediction_errors_total` | counter | - |
| `cardiorisk_model_info` | gauge | version, model_type |

Unknown URLs are grouped under `path="unmatched"` so scanners cannot create unbounded series.

### Prometheus + Grafana

```bash
make monitoring     # scripts/deploy_monitoring.sh -> namespace "monitoring"
make grafana        # http://localhost:3000  (anonymous read-only; dashboard "CardioRisk API")
make prometheus     # http://localhost:9090/targets
make load           # generate traffic so the panels fill up
```

- **Prometheus** discovers API pods through the Kubernetes API using their
  `prometheus.io/*` annotations (each pod is a separate target, which is why the container runs a
  single uvicorn worker), and scrapes cAdvisor through the API server for pod CPU / memory.
- **Alert rules** (`k8s/monitoring/prometheus/alerts.yml`): API down, 5xx rate > 2%, p95 latency
  > 250 ms, more than 20% invalid requests, and a **prediction-drift** alert when the live share
  of positive predictions leaves the 15-75% band around the 46% training baseline.
- **Grafana** is provisioned entirely from code - data source, dashboard provider and the dashboard
  generated by `scripts/build_dashboard.py`. The admin password is a random secret created at deploy
  time (`kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.password}' | base64 -d`).

Dashboard rows: **service health** (pods up, req/s, p95 latency, 5xx rate, rejected inputs, model
version), **traffic & latency** (requests by route/status, p50/p95/p99 vs model-only latency),
**model behaviour** (predictions by risk band, positive rate vs training baseline, probability
distribution) and **resources** (CPU, memory, in-flight requests per pod).

### One-shot run

```bash
make all        # download -> clean -> figures -> train -> tests
make predict
```

## Tests

```bash
pytest -v --cov        # 60 tests: data, features, model, evaluation, downloader, API
```
