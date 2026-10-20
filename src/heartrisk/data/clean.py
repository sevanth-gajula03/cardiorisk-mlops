"""Clean the raw Cleveland file and write analysis-ready CSVs.

Steps:
  1. Load with proper column names; '?' becomes NaN.
  2. Drop exact duplicate rows.
  3. Flag readings outside clinically plausible ranges as missing.
  4. Impute: median for continuous columns, mode for coded columns.
     (Only a handful of rows are affected, so dropping them would waste data.)
  5. Collapse the 0-4 diagnosis into a binary target (0 = no disease, 1 = disease).
  6. Cast coded columns to integers and write heart_clean.csv.
  7. One-hot encode the nominal columns and write heart_encoded.csv.

A JSON data quality report records what was changed at each step.

Usage:
    python -m heartrisk.data.clean
"""

import json
import logging

import pandas as pd

from heartrisk.config import load_config, resolve

log = logging.getLogger(__name__)

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Human-readable labels for the coded columns (from heart-disease.names).
CATEGORY_LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical_angina", 2: "atypical_angina", 3: "non_anginal", 4: "asymptomatic"},
    "fbs": {0: "<=120", 1: ">120"},
    "restecg": {0: "normal", 1: "st_t_abnormal", 2: "lv_hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed_defect", 7: "reversible_defect"},
}


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")


def flag_out_of_range(df: pd.DataFrame, ranges: dict) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    flagged = {}
    for col, (low, high) in ranges.items():
        mask = df[col].notna() & ~df[col].between(low, high)
        flagged[col] = int(mask.sum())
        df.loc[mask, col] = pd.NA
    return df, flagged


def impute(df: pd.DataFrame, numeric: list, coded: list) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    filled = {}
    for col in numeric + coded:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        value = df[col].median() if col in numeric else df[col].mode().iloc[0]
        df[col] = df[col].fillna(value)
        filled[col] = {"rows": n_missing, "value": float(value)}
    return df, filled


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    cols = cfg["columns"]
    numeric = cols["numeric"]
    coded = cols["ordinal"] + cols["binary"] + cols["nominal"]
    report = {"rows_in": len(df), "missing_before": df.isna().sum()[lambda s: s > 0].to_dict()}

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)

    df, report["out_of_range_flagged"] = flag_out_of_range(df, cfg["valid_ranges"])
    df, report["imputed"] = impute(df, numeric, coded)

    df[cols["target"]] = (df.pop("num") > 0).astype(int)
    df[coded] = df[coded].astype(int)

    report["rows_out"] = len(df)
    report["class_counts"] = df[cols["target"]].value_counts().sort_index().to_dict()
    report["missing_after"] = int(df.isna().sum().sum())
    return df, report


def one_hot(df: pd.DataFrame, nominal: list) -> pd.DataFrame:
    readable = df.copy()
    for col in nominal:
        readable[col] = readable[col].map(CATEGORY_LABELS[col])
    return pd.get_dummies(readable, columns=nominal, prefix=nominal, dtype=int)


def with_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the clean frame with codes replaced by readable labels (for plots)."""
    out = df.copy()
    for col, mapping in CATEGORY_LABELS.items():
        out[col] = out[col].map(mapping)
    return out


def run() -> pd.DataFrame:
    cfg = load_config()
    paths = cfg["data"]

    raw = load_raw(resolve(paths["raw_file"]))
    clean_df, report = clean(raw, cfg)
    encoded = one_hot(clean_df, cfg["columns"]["nominal"])

    clean_path = resolve(paths["clean_file"])
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_path, index=False)
    encoded.to_csv(resolve(paths["encoded_file"]), index=False)

    report_path = resolve(paths["quality_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=int))

    log.info("Clean data: %d rows x %d cols -> %s", *clean_df.shape, clean_path)
    log.info("Encoded data: %d cols", encoded.shape[1])
    log.info("Imputed: %s", report["imputed"])
    return clean_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run()


if __name__ == "__main__":
    main()
