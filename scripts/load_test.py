"""Send realistic traffic to the API (for dashboards, autoscaling and the demo video).

Patients are sampled from the cleaned dataset with small random jitter, and a
small share of deliberately invalid requests is mixed in so error panels show
something too. Standard library only.

    python scripts/load_test.py --url http://127.0.0.1:8090 --rps 20 --duration 120
    python scripts/load_test.py --url http://localhost:8081 --host cardiorisk.local
"""

import argparse
import csv
import json
import random
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def load_patients() -> list[dict]:
    path = ROOT / "data" / "processed" / "heart_clean.csv"
    if not path.exists():
        return json.loads((ROOT / "samples" / "patients.json").read_text())
    with open(path) as fh:
        return [{k: float(row[k]) for k in FEATURES} for row in csv.DictReader(fh)]


def jitter(patient: dict) -> dict:
    p = dict(patient)
    p["age"] = int(min(90, max(25, p["age"] + random.randint(-3, 3))))
    p["trestbps"] = min(200, max(90, p["trestbps"] + random.uniform(-8, 8)))
    p["chol"] = min(550, max(120, p["chol"] + random.uniform(-20, 20)))
    p["thalach"] = min(200, max(70, p["thalach"] + random.uniform(-10, 10)))
    for k in ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"):
        p[k] = int(p[k])
    return p


def send(url: str, host: str | None, body: dict, stats: Counter, lock: threading.Lock) -> None:
    headers = {"content-type": "application/json"}
    if host:
        headers["Host"] = host
    req = urllib.request.Request(f"{url}/predict", data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            key = f"200 {result['risk_band']}"
    except urllib.error.HTTPError as exc:
        key = str(exc.code)
    except Exception:
        key = "error"
    with lock:
        stats[key] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="CardioRisk load generator")
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--host", default=None, help="Host header, e.g. cardiorisk.local for the ingress")
    parser.add_argument("--rps", type=float, default=10)
    parser.add_argument("--duration", type=int, default=60, help="seconds")
    parser.add_argument("--invalid-share", type=float, default=0.05)
    args = parser.parse_args()

    patients = load_patients()
    stats, lock = Counter(), threading.Lock()
    interval = 1.0 / args.rps
    deadline = time.time() + args.duration
    print(f"Sending ~{args.rps} req/s to {args.url} for {args.duration}s")

    with ThreadPoolExecutor(max_workers=32) as pool:
        next_report = time.time() + 10
        while time.time() < deadline:
            body = jitter(random.choice(patients))
            if random.random() < args.invalid_share:
                body["age"] = 250
            pool.submit(send, args.url, args.host, body, stats, lock)
            if time.time() >= next_report:
                with lock:
                    print(f"  {sum(stats.values())} sent: {dict(stats)}")
                next_report += 10
            time.sleep(interval)

    print(f"Finished: {sum(stats.values())} requests -> {dict(stats)}")


if __name__ == "__main__":
    main()
