# Screenshots

Evidence used in the report (`docs/CardioRisk_MLOps_Report.pdf`).

| # | File | Shows |
|---|---|---|
| 01 | `01_eda_notebook.png` | executed EDA notebook (any figure cell) |
| 02 | `02_training_terminal.png` | `make train` output with the six runs and the champion |
| 03 | `03_mlflow_runs.jpg` | MLflow session with nested runs and the registered model |
| 04 | `04_mlflow_metric_comparison.jpg` | MLflow chart view comparing `cv_roc_auc` across runs |
| 05 | `05_api_swagger.jpg` | Swagger UI of the deployed API |
| 06 | `06_api_predict_response.jpg` | live `/predict` response from the Kubernetes deployment |
| 07 | `07_ci_pipeline_success.jpg` | green GitHub Actions run (all five jobs) |
| 08 | `08_grafana_dashboard.jpg` | Grafana dashboard under load |
| 09 | `09_ci_failure_pr.jpg` | pipeline failing on a pull request with a broken test |
| 10 | `10_ci_summary.png` | run summary page: coverage, model table, quality gate, image |
| 11 | `11_ci_artifacts.png` | artifacts list of a CI run |
| 12 | `12_kubectl_status.png` | `make k8s-status` (deployment, pods, service, ingress, HPA, PDB) |
| 13 | `13_docker_build_run.png` | `make docker-build docker-run smoke` |
| 14 | `14_prometheus_targets.png` | Prometheus `/targets` with both API pods and cAdvisor up |
| 15 | `15_hpa_scaling.png` | `kubectl get hpa -w` while `make load` runs |
