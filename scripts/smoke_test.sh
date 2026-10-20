#!/usr/bin/env bash
# End-to-end check of a running CardioRisk API.
#   scripts/smoke_test.sh                      # defaults to http://localhost:8080
#   HOST_HEADER=cardiorisk.local scripts/smoke_test.sh http://localhost:8081   # via ingress
# Exits non-zero on the first failed check, so it can gate CI and deployments.
set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
CURL=(curl -s)
[[ -n "${HOST_HEADER:-}" ]] && CURL+=(-H "Host: ${HOST_HEADER}")

check() {
  local name="$1" expected="$2" actual="$3"
  if [[ "$actual" == "$expected" ]]; then
    printf '  [ok]   %s\n' "$name"; PASS=$((PASS + 1))
  else
    printf '  [FAIL] %s (expected %s, got %s)\n' "$name" "$expected" "$actual"; exit 1
  fi
}

echo "Smoke testing ${BASE_URL}"

for _ in $(seq 1 30); do
  "${CURL[@]}" -f "${BASE_URL}/ready" >/dev/null 2>&1 && break
  sleep 2
done

check "GET /health"  200 "$("${CURL[@]}" -o /dev/null -w '%{http_code}' "${BASE_URL}/health")"
check "GET /ready"   200 "$("${CURL[@]}" -o /dev/null -w '%{http_code}' "${BASE_URL}/ready")"

high_risk='{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
body=$("${CURL[@]}" -X POST "${BASE_URL}/predict" -H 'content-type: application/json' -d "$high_risk")
echo "  /predict -> ${body}"
check "high-risk patient predicted as disease" 1 \
  "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["prediction"])' "$body")"

batch=$(python3 -c 'import json,sys; print(json.dumps({"patients": json.load(open(sys.argv[1]))}))' \
  "${ROOT}/samples/patients.json")
check "POST /predict/batch" 200 "$("${CURL[@]}" -o /dev/null -w '%{http_code}' -X POST \
  "${BASE_URL}/predict/batch" -H 'content-type: application/json' -d "$batch")"

check "invalid input rejected" 422 "$("${CURL[@]}" -o /dev/null -w '%{http_code}' -X POST \
  "${BASE_URL}/predict" -H 'content-type: application/json' -d '{"age": 300}')"

check "metrics exposed" 1 "$("${CURL[@]}" "${BASE_URL}/metrics" | grep -c '^cardiorisk_model_info')"

echo "All ${PASS} checks passed."
