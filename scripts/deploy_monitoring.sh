#!/usr/bin/env bash
# Deploy Prometheus + Grafana into Minikube and print how to open them.
#   scripts/deploy_monitoring.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NS=monitoring

step() { printf '\n==> %s\n' "$*"; }

step "Rendering Grafana dashboard"
python3 scripts/build_dashboard.py

step "Applying monitoring stack"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
# Create the Grafana admin password once; keep it on later runs.
if ! kubectl -n "$NS" get secret grafana-admin >/dev/null 2>&1; then
  kubectl -n "$NS" create secret generic grafana-admin \
    --from-literal=password="$(openssl rand -hex 12)" >/dev/null
fi
kubectl apply -k k8s/monitoring

step "Waiting for pods"
kubectl -n "$NS" rollout status deployment/prometheus --timeout=180s
kubectl -n "$NS" rollout status deployment/grafana --timeout=180s
kubectl -n "$NS" get pods,svc

cat <<EOF

Open them with port-forwards (each in its own terminal):
  kubectl -n $NS port-forward svc/prometheus 9090:9090   -> http://localhost:9090/targets
  kubectl -n $NS port-forward svc/grafana 3000:3000      -> http://localhost:3000 (anonymous viewer)

Grafana admin password:
  kubectl -n $NS get secret grafana-admin -o jsonpath='{.data.password}' | base64 -d; echo
EOF
