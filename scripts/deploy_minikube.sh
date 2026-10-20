#!/usr/bin/env bash
# Build the API image, load it into Minikube and roll it out.
#   scripts/deploy_minikube.sh            # build + deploy + smoke test through the ingress
#   SKIP_BUILD=1 scripts/deploy_minikube.sh
#
# The image is tagged with the model version from models/metadata.json, so every
# retrained model produces a new tag and a proper rolling update.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NAMESPACE=cardiorisk
OVERLAY=k8s/overlays/minikube
IMAGE=cardiorisk-api
TAG="$(python3 -c "import json; print(json.load(open('models/metadata.json'))['model_version'])")"
INGRESS_PORT="${INGRESS_PORT:-8081}"

step() { printf '\n==> %s\n' "$*"; }

step "Checking cluster"
minikube status >/dev/null || { echo "Minikube is not running: minikube start"; exit 1; }
kubectl config use-context minikube >/dev/null

if [[ -z "${SKIP_BUILD:-}" ]]; then
  step "Building ${IMAGE}:${TAG}"
  docker build -t "${IMAGE}:${TAG}" \
    --build-arg MODEL_VERSION="${TAG}" \
    --build-arg GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo local)" .

  step "Loading image into Minikube"
  minikube image load "${IMAGE}:${TAG}"
fi

step "Pinning image tag in ${OVERLAY}"
sed -i.bak -E "s/^( *newTag: ).*/\1\"${TAG}\"/" "${OVERLAY}/kustomization.yaml" && rm -f "${OVERLAY}/kustomization.yaml.bak"

step "Applying manifests"
kubectl apply -k "${OVERLAY}"
kubectl -n "${NAMESPACE}" annotate deployment/cardiorisk-api \
  kubernetes.io/change-cause="deploy model ${TAG}" --overwrite >/dev/null

step "Waiting for rollout"
kubectl -n "${NAMESPACE}" rollout status deployment/cardiorisk-api --timeout=180s
kubectl -n "${NAMESPACE}" get deploy,pods,svc,ingress,hpa -o wide

step "Smoke test through the ingress (Host: cardiorisk.local)"
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller "${INGRESS_PORT}:80" >/dev/null 2>&1 &
PF_PID=$!
trap 'kill ${PF_PID} 2>/dev/null || true' EXIT
sleep 3
HOST_HEADER=cardiorisk.local scripts/smoke_test.sh "http://localhost:${INGRESS_PORT}"

step "Done - model ${TAG} is live"
cat <<EOF
Access options:
  Ingress:       kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller ${INGRESS_PORT}:80
                 curl -H 'Host: cardiorisk.local' http://localhost:${INGRESS_PORT}/health
  LoadBalancer:  minikube tunnel   (then use the EXTERNAL-IP of svc/cardiorisk-api on port 8090)
  Swagger UI:    kubectl -n ${NAMESPACE} port-forward svc/cardiorisk-api 8090:8090 -> http://localhost:8090/docs
EOF
