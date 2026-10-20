"""Metrics and diagnostic plots for a fitted classifier."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from heartrisk.config import load_config

# Scorers used by GridSearchCV; keys double as MLflow metric names. Strongly
# regularised grid points can predict a single class, so precision uses zero_division=0.
CV_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def _colors() -> dict:
    p = load_config()["eda"]["palette"]
    return {"healthy": p["healthy"], "disease": p["disease"], "accent": p["accent"]}


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
    }


def feature_importance(pipeline) -> pd.Series:
    """Signed coefficients for linear models, impurity importances for trees."""
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "coef_"):
        values = model.coef_.ravel()
    else:
        values = model.feature_importances_
    return pd.Series(values, index=names).sort_values(key=np.abs, ascending=False)


def plot_confusion(y_true, y_prob, title: str, threshold: float = 0.5) -> plt.Figure:
    c = _colors()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("cr", ["#ffffff", c["accent"]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (np.asarray(y_prob) >= threshold).astype(int),
        display_labels=["No disease", "Disease"], cmap=cmap, colorbar=False, ax=ax)
    ax.grid(False)
    ax.set_title(title, fontsize=10, fontweight="bold")
    return fig


def plot_roc_pr(y_true, y_prob, title: str) -> plt.Figure:
    c = _colors()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, color=c["disease"], lw=2, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax1.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    ax1.legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ax2.plot(rec, prec, color=c["accent"], lw=2,
             label=f"AP = {average_precision_score(y_true, y_prob):.3f}")
    ax2.axhline(np.mean(y_true), ls="--", color="grey", lw=1, label="baseline")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax2.legend(loc="lower left")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_calibration(y_true, y_prob, title: str) -> plt.Figure:
    """Do predicted probabilities match observed rates? Matters because the API returns them."""
    c = _colors()
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, marker="o", color=c["disease"], lw=2, label="model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed disease rate",
           xlim=(0, 1), ylim=(0, 1), title=title)
    ax.legend()
    return fig


def plot_importance(importance: pd.Series, title: str, top: int = 15) -> plt.Figure:
    c = _colors()
    top_imp = importance.head(top)[::-1]
    colors = [c["disease"] if v > 0 else c["healthy"] for v in top_imp.values]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(top_imp.index, top_imp.values, color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Coefficient (red raises risk)" if (importance < 0).any() else "Importance")
    return fig


def plot_model_comparison(summary: pd.DataFrame) -> plt.Figure:
    """CV ROC-AUC (mean +/- std) and test ROC-AUC for every run in the session."""
    c = _colors()
    s = summary.sort_values("cv_roc_auc")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(s) + 1.5))
    ax.errorbar(s["cv_roc_auc"], y, xerr=s["cv_roc_auc_std"], fmt="o", color=c["accent"],
                capsize=4, label="CV mean +/- std")
    ax.scatter(s["test_roc_auc"], y, marker="D", color=c["disease"], label="hold-out test", zorder=3)
    ax.set_yticks(y, s["run_name"])
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Model comparison", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def plot_roc_overlay(curves: dict, y_true) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} ({roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC on hold-out test set")
    ax.legend(fontsize=7, loc="lower right")
    return fig
