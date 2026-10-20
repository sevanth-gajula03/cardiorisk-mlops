# Screenshots

Evidence for the report (`docs/CardioRisk_MLOps_Report.pdf`), captured from the running system.

| # | File | Shows |
|---|---|---|
| 01 | `01_eda_notebook.png` | executed EDA notebook - disease rate by category |
| 02 | `02_training_terminal.png` | training session: six tuned runs and the selected champion |
| 03 | `03_mlflow_runs.jpg` | MLflow session with nested runs and the registered model |
| 04 | `04_mlflow_metric_comparison.jpg` | MLflow chart view comparing `cv_roc_auc` across runs |
| 05 | `05_api_swagger.jpg` | Swagger UI of the deployed API |
| 06 | `06_api_predict_response.jpg` | live `/predict` response from the Kubernetes deployment |
| 07 | `07_ci_pipeline_success.jpg` | green GitHub Actions run (all five jobs) |
| 08 | `08_grafana_dashboard.jpg` | Grafana dashboard under load |
| 09 | `09_ci_failure_pr.jpg` | pipeline failing on a pull request with a broken test |
| 10 | `10_ci_artifacts.png` | CI run with its uploaded artifacts (model, reports, MLflow runs, test reports, logs) |
| 12 | `12_kubectl_status.png` | deployment, pods, LoadBalancer service, Ingress, HPA, PDB and rollout history |
| 13 | `13_docker_build_run.png` | container started locally and the 6-check smoke test passing |
| 14 | `14_prometheus_targets.png` | Prometheus targets: both API pods, cAdvisor and Prometheus up |
| 15 | `15_hpa_scaling.png` | autoscaler at 112% CPU scaling to 4 pods, plus JSON prediction logs |
| 16 | `16_self_healing.png` | deleted pods replaced automatically |
| 17 | `17_prometheus_alert_rules.png` | loaded alert rules (availability, errors, latency, drift, invalid input) |
