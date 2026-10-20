"""Unit tests for data loading and cleaning (no network needed)."""

import io

import numpy as np
import pandas as pd
import pytest

from heartrisk.config import load_config
from heartrisk.data import clean
from heartrisk.data.download import sha256_of

RAW_SAMPLE = """\
63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1
37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,?,3.0,0
41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,0.0,?,0
56.0,1.0,2.0,120.0,999.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,3.0,4
63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
"""


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def raw():
    return clean.load_raw(io.StringIO(RAW_SAMPLE))


def test_load_raw_parses_question_marks_as_missing(raw):
    assert list(raw.columns) == clean.RAW_COLUMNS
    assert raw["ca"].isna().sum() == 1
    assert raw["thal"].isna().sum() == 1


def test_clean_leaves_no_missing_values(raw, cfg):
    df, report = clean.clean(raw, cfg)
    assert df.isna().sum().sum() == 0
    assert report["missing_after"] == 0


def test_clean_removes_duplicates(raw, cfg):
    df, report = clean.clean(raw, cfg)
    assert report["duplicates_removed"] == 1
    assert len(df) == len(raw) - 1


def test_out_of_range_values_are_flagged_and_imputed(raw, cfg):
    df, report = clean.clean(raw, cfg)
    assert report["out_of_range_flagged"]["chol"] == 1
    assert df["chol"].max() <= cfg["valid_ranges"]["chol"][1]


def test_target_is_binary(raw, cfg):
    df, _ = clean.clean(raw, cfg)
    assert "num" not in df.columns
    assert set(df["target"].unique()) <= {0, 1}
    # num = 2, 1 and 4 all become disease
    assert df["target"].sum() == 3


def test_coded_columns_are_integers_with_valid_codes(raw, cfg):
    df, _ = clean.clean(raw, cfg)
    for col in cfg["columns"]["nominal"] + cfg["columns"]["binary"]:
        assert np.issubdtype(df[col].dtype, np.integer)
        assert set(df[col].unique()) <= set(clean.CATEGORY_LABELS[col])


def test_missing_codes_use_mode_not_median(cfg):
    # With an even 3/7 split the median would be 5.0, which is not a valid thal code.
    frame = clean.load_raw(io.StringIO(RAW_SAMPLE)).iloc[:5].copy()
    frame["thal"] = [3.0, 7.0, 3.0, 7.0, np.nan]
    df, _ = clean.clean(frame, cfg)
    assert df["thal"].iloc[-1] in clean.CATEGORY_LABELS["thal"]


def test_one_hot_creates_readable_columns(raw, cfg):
    df, _ = clean.clean(raw, cfg)
    encoded = clean.one_hot(df, cfg["columns"]["nominal"])
    assert "cp_asymptomatic" in encoded.columns
    assert "thal_reversible_defect" in encoded.columns
    for col in cfg["columns"]["nominal"]:
        assert col not in encoded.columns
    one_hot_cp = encoded.filter(like="cp_")
    assert (one_hot_cp.sum(axis=1) == 1).all()


def test_with_labels_maps_codes():
    df = pd.DataFrame({c: [k] for c, k in [("sex", 1), ("cp", 4), ("fbs", 0), ("restecg", 0),
                                          ("exang", 1), ("slope", 2), ("thal", 7)]})
    labelled = clean.with_labels(df)
    assert labelled.loc[0, "cp"] == "asymptomatic"
    assert labelled.loc[0, "thal"] == "reversible_defect"


def test_sha256_is_stable():
    assert sha256_of(b"heart") == sha256_of(b"heart")
    assert sha256_of(b"heart") != sha256_of(b"heart!")
