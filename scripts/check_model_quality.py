"""Fail (exit 1) if the packaged model misses the minimum metrics in params.yaml.

    python scripts/check_model_quality.py [--summary path]

With --summary the result table is appended as markdown (used for the GitHub
Actions job summary).
"""

import argparse
import json
import sys

from heartrisk.config import load_config, resolve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", help="append a markdown table to this file")
    args = parser.parse_args()

    cfg = load_config()["training"]
    gate = cfg["quality_gate"]
    metadata = json.loads((resolve(cfg["model_dir"]) / "metadata.json").read_text())
    metrics = metadata["metrics"]

    rows, failed = [], False
    for name, minimum in gate.items():
        value = metrics[name]
        ok = value >= minimum
        failed |= not ok
        rows.append(f"| {name} | {value:.3f} | >= {minimum:.2f} | {'pass' if ok else 'FAIL'} |")
        print(f"{'PASS' if ok else 'FAIL'}  {name:<14} {value:.4f}  (minimum {minimum})")

    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write(f"### Model quality gate - {metadata['model_type']} v{metadata['model_version']}\n\n")
            fh.write("| Metric | Value | Required | Result |\n|---|---|---|---|\n")
            fh.write("\n".join(rows) + "\n\n")

    if failed:
        print("Model quality gate failed - refusing to ship this model.")
        return 1
    print("Model quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
