"""Load the packaged model and score patient records.

Usage:
    python -m heartrisk.models.predict --input samples/patients.json
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from heartrisk.config import load_config, resolve
from heartrisk.features import RAW_FEATURES

log = logging.getLogger(__name__)

# Probability cut-offs for the human-friendly risk band returned with each prediction.
RISK_BANDS = [(0.35, "low"), (0.6, "moderate"), (1.01, "high")]


def risk_band(probability: float) -> str:
    for upper, name in RISK_BANDS:
        if probability < upper:
            return name
    return "high"


class HeartRiskModel:
    """Thin wrapper around the saved sklearn pipeline and its metadata."""

    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = metadata.get("decision_threshold", 0.5)

    @classmethod
    def load(cls, model_dir: str | Path | None = None) -> "HeartRiskModel":
        model_dir = Path(model_dir) if model_dir else resolve(load_config()["training"]["model_dir"])
        pipeline = joblib.load(model_dir / "model.joblib")
        metadata = json.loads((model_dir / "metadata.json").read_text())
        return cls(pipeline, metadata)

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unknown")

    def _to_frame(self, records: list[dict]) -> pd.DataFrame:
        missing = [f for f in RAW_FEATURES if any(f not in r for r in records)]
        if missing:
            raise ValueError(f"Missing required fields: {sorted(set(missing))}")
        return pd.DataFrame(records)[RAW_FEATURES].astype(float)

    def explain(self, frame: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Per-patient top factors for linear models: coefficient x transformed value.

        Values are standardised, so a factor's weight is how far it moves this
        patient's log-odds compared with an average patient. Tree models return [].
        """
        model = self.pipeline.named_steps["model"]
        if not hasattr(model, "coef_"):
            return [[] for _ in range(len(frame))]
        transformed = self.pipeline[:-1].transform(frame)
        contributions = transformed * model.coef_.ravel()
        explanations = []
        for _, row in contributions.iterrows():
            strongest = row[row != 0].abs().sort_values(ascending=False).head(top).index
            explanations.append([
                {"feature": name,
                 "effect": "raises_risk" if row[name] > 0 else "lowers_risk",
                 "weight": round(float(row[name]), 3)}
                for name in strongest
            ])
        return explanations

    def predict(self, records: list[dict], explain: bool = False) -> list[dict]:
        frame = self._to_frame(records)
        probs = self.pipeline.predict_proba(frame)[:, 1]
        factors = self.explain(frame) if explain else [None] * len(frame)
        results = []
        for p, why in zip(probs, factors, strict=True):
            p = float(p)
            label = int(p >= self.threshold)
            result = {
                "prediction": label,
                "label": "heart_disease" if label else "no_heart_disease",
                "probability": round(p, 4),
                "confidence": round(p if label else 1 - p, 4),
                "risk_band": risk_band(p),
            }
            if why is not None:
                result["top_factors"] = why
            results.append(result)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="JSON file with one record or a list")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    payload = json.loads(Path(args.input).read_text())
    records = payload if isinstance(payload, list) else [payload]
    model = HeartRiskModel.load(args.model_dir)
    log.info("Loaded model version %s (%s)", model.version, model.metadata["model_type"])
    for record, result in zip(records, model.predict(records, explain=True), strict=True):
        print(json.dumps({"input": record, "output": result}))


if __name__ == "__main__":
    main()
