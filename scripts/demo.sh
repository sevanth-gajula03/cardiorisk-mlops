#!/usr/bin/env bash
# Walk-through of the whole pipeline for the demo video.
# Each step pauses so it can be narrated; press Enter to continue.
#   scripts/demo.sh            # assumes colima + minikube are running
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

say()   { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
pause() { read -r -p $'\033[2m   [enter] to continue\033[0m ' _; }
run()   { printf '\033[1;33m$ %s\033[0m\n' "$*"; eval "$@"; }

say "1. Data: download from UCI, clean, record checksum"
run "$PY -m heartrisk.data.download --force"
run "$PY -m heartrisk.data.clean"
run "cat data/raw/MANIFEST.json"
pause

say "2. Train 3 models x 2 feature sets, tracked in MLflow"
run "$PY -m heartrisk.models.train"
run "cat reports/model_comparison.md | head -16"
pause

say "3. Quality gate + tests + lint (what CI runs)"
run "$PY scripts/check_model_quality.py"
run "ruff check ."
run "$PY -m pytest -q"
pause

say "4. Package and run the container locally"
run "make docker-build PYTHON=$PY"
run "make docker-run"
run "scripts/smoke_test.sh http://localhost:8080"
pause

say "5. Deploy to Minikube (rolling update, smoke test through the ingress)"
run "scripts/deploy_minikube.sh"
run "kubectl -n cardiorisk get deploy,pods,svc,ingress,hpa"
pause

say "6. Monitoring: Prometheus + Grafana, then generate traffic"
run "scripts/deploy_monitoring.sh"
echo "Open Grafana: kubectl -n monitoring port-forward svc/grafana 3000:3000  -> http://localhost:3000"
echo "Expose the API: minikube tunnel  (or: kubectl -n cardiorisk port-forward svc/cardiorisk-api 8090:8090)"
pause
run "$PY scripts/load_test.py --url http://127.0.0.1:8090 --rps 20 --duration 90"
run "kubectl -n cardiorisk get hpa"
run "kubectl -n cardiorisk logs -l app.kubernetes.io/name=cardiorisk-api --tail=3"

say "Done. CI runs: https://github.com/sevanth-gajula03/cardiorisk-mlops/actions"
