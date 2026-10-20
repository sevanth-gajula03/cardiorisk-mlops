# Demo video outline (about 5 minutes)

Before recording: `colima start`, `minikube start`, `minikube tunnel` in a spare terminal,
and Grafana/MLflow port-forwards ready (`make grafana`, `make mlflow-ui`).
Run `scripts/demo.sh` in the main terminal and talk over each pause.

| Time | Show | Say |
|---|---|---|
| 0:00 | README + architecture diagram | Problem, dataset, and the five stages of the system |
| 0:30 | demo step 1 - download and clean, `MANIFEST.json` | Checksum makes the data traceable; 6 missing values imputed with the mode |
| 0:50 | `notebooks/01_eda.ipynb` (disease-rate and age-band figures) | Strongest signals, and why they led to `hr_reserve` and `age_x_male` |
| 1:20 | demo step 2 - training log, `reports/model_comparison.md` | 3 models x 2 feature sets, CV-based selection, engineered features helped all three |
| 1:50 | MLflow UI - nested runs, chart view, registered model | What is logged per run; champion alias |
| 2:20 | demo step 3 - quality gate, ruff, pytest | Same checks the CI runs |
| 2:40 | GitHub Actions - green run, summary page, artifacts, failing PR | Job chain, quality gate, pipeline stops on a broken test |
| 3:20 | demo step 4 - docker build/run + smoke test | Non-root image, one worker per pod, 6 checks |
| 3:40 | demo step 5 - deploy, `kubectl get ...`, Swagger `/predict` | Rolling update tagged by model version, ingress + load balancer, explanation in the response |
| 4:20 | demo step 6 - load test, Grafana, `kubectl get hpa` | Latency, risk bands, drift panel, autoscaling to 3 pods, JSON logs with request ids |
| 4:50 | wrap-up | What I would add next (remote MLflow, data-drift checks, managed cluster) |
