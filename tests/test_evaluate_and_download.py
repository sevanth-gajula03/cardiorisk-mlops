"""Tests for evaluation plots and the dataset downloader (network is mocked)."""

import io
import json
import urllib.error
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from heartrisk.data import download
from heartrisk.models import evaluate

matplotlib.use("Agg")

Y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1])
P = np.array([0.1, 0.3, 0.8, 0.7, 0.2, 0.9, 0.6, 0.4, 0.55, 0.35, 0.75, 0.65])


@pytest.mark.parametrize("plot", [evaluate.plot_confusion, evaluate.plot_roc_pr, evaluate.plot_calibration])
def test_diagnostic_plots_return_figures(plot):
    fig = plot(Y, P, "test")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig.axes


def test_importance_and_comparison_plots():
    imp = pd.Series([0.8, -0.5, 0.1], index=["ca", "thalach", "chol"])
    assert evaluate.plot_importance(imp, "imp").axes
    summary = pd.DataFrame({"run_name": ["a", "b"], "cv_roc_auc": [0.9, 0.88],
                            "cv_roc_auc_std": [0.01, 0.02], "test_roc_auc": [0.93, 0.95]})
    assert evaluate.plot_model_comparison(summary).axes
    assert evaluate.plot_roc_overlay({"a": P, "b": P[::-1]}, Y).axes


@pytest.fixture
def fake_paths(tmp_path, monkeypatch):
    cfg = {"data": {
        "source_url": "https://example.test/archive.zip",
        "fallback_url": "https://example.test/file.data",
        "member": "processed.cleveland.data",
        "raw_file": str(tmp_path / "raw.data"),
        "manifest_file": str(tmp_path / "MANIFEST.json"),
    }}
    monkeypatch.setattr(download, "load_config", lambda: cfg)
    monkeypatch.setattr(download, "resolve", Path)
    return tmp_path


def _zip_with(member: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, payload)
    return buf.getvalue()


def test_download_from_archive_writes_manifest(fake_paths, monkeypatch):
    payload = b"63.0,1.0\n67.0,1.0"
    monkeypatch.setattr(download, "fetch", lambda url: _zip_with("processed.cleveland.data", payload))
    manifest = download.download(force=True)
    assert (fake_paths / "raw.data").read_bytes() == payload
    assert manifest["rows"] == 2
    assert manifest["sha256"] == download.sha256_of(payload)
    assert json.loads((fake_paths / "MANIFEST.json").read_text())["source"].endswith("archive.zip")


def test_download_falls_back_to_direct_link(fake_paths, monkeypatch):
    def fake_fetch(url):
        if url.endswith(".zip"):
            raise urllib.error.URLError("archive offline")
        return b"1,2,3"

    monkeypatch.setattr(download, "fetch", fake_fetch)
    manifest = download.download(force=True)
    assert manifest["source"].endswith("file.data")


def test_download_skips_when_file_exists(fake_paths, monkeypatch):
    (fake_paths / "raw.data").write_bytes(b"cached")
    (fake_paths / "MANIFEST.json").write_text('{"rows": 1}')
    monkeypatch.setattr(download, "fetch", lambda url: pytest.fail("should not download"))
    assert download.download() == {"rows": 1}
