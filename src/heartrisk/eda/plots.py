"""Exploratory plots for the heart disease data.

Every function takes a DataFrame and returns a matplotlib Figure so the same
code works in the notebook and from the command line.

Usage:
    python -m heartrisk.eda.plots
"""

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heartrisk.config import load_config, resolve
from heartrisk.data.clean import load_raw, with_labels

log = logging.getLogger(__name__)
# Category codes such as ca = "0".."3" are plotted as labels on purpose.
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

PRETTY = {
    "age": "Age (years)",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "thalach": "Max heart rate (bpm)",
    "oldpeak": "ST depression (oldpeak)",
    "ca": "Major vessels coloured (ca)",
    "cp": "Chest pain type",
    "thal": "Thallium stress result",
    "slope": "ST segment slope",
    "exang": "Exercise-induced angina",
    "sex": "Sex",
    "restecg": "Resting ECG",
    "fbs": "Fasting blood sugar",
}


def _palette() -> dict:
    p = load_config()["eda"]["palette"]
    return {0: p["healthy"], 1: p["disease"], "accent": p["accent"]}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })


def class_balance(df: pd.DataFrame) -> plt.Figure:
    pal = _palette()
    counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["No disease (0)", "Disease (1)"], counts.values, color=[pal[0], pal[1]], width=0.55)
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n}\n({n / counts.sum():.0%})", (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("Class balance of the target")
    return fig


def missing_values(raw: pd.DataFrame) -> plt.Figure:
    pal = _palette()
    missing = raw.isna().sum()
    missing = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.barh(missing.index, missing.values, color=pal["accent"])
    for i, n in enumerate(missing.values):
        ax.text(n + 0.1, i, f"{n} rows ({n / len(raw):.1%})", va="center", fontsize=9)
    ax.set_xlim(0, missing.max() * 2)
    ax.set_xlabel("Missing values in raw file")
    ax.set_title("Missing values before cleaning")
    return fig


def numeric_histograms(df: pd.DataFrame, numeric: list) -> plt.Figure:
    pal = _palette()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, numeric):
        sns.histplot(data=df, x=col, hue="target", palette=pal, kde=True, element="step",
                     stat="density", common_norm=False, alpha=0.3, ax=ax)
        ax.set_xlabel(PRETTY.get(col, col))
        ax.get_legend().set_title("Target")
    # Last panel: overall age distribution with mean/median markers.
    ax = axes.flat[-1]
    sns.histplot(df["age"], bins=20, color=pal["accent"], ax=ax)
    ax.axvline(df["age"].mean(), color=pal[1], ls="--", label=f"mean {df['age'].mean():.1f}")
    ax.axvline(df["age"].median(), color=pal[0], ls=":", label=f"median {df['age'].median():.0f}")
    ax.legend(fontsize=8)
    ax.set_xlabel("Age (all patients)")
    fig.suptitle("Distribution of continuous features by target", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def numeric_boxplots(df: pd.DataFrame, numeric: list) -> plt.Figure:
    pal = _palette()
    fig, axes = plt.subplots(1, len(numeric), figsize=(15, 3.8))
    for ax, col in zip(axes, numeric):
        sns.boxplot(data=df, x="target", y=col, hue="target", palette=pal, legend=False,
                    width=0.5, fliersize=3, ax=ax)
        ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_xlabel("Heart disease")
        ax.set_ylabel("")
        ax.set_title(PRETTY.get(col, col), fontsize=10)
    fig.suptitle("Spread and outliers of continuous features", fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.5, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.grid(False)
    ax.set_title("Correlation matrix (lower triangle)")
    return fig


def target_correlation(df: pd.DataFrame) -> plt.Figure:
    pal = _palette()
    corr = df.corr(numeric_only=True)["target"].drop("target").sort_values()
    colors = [pal[1] if v > 0 else pal[0] for v in corr.values]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    for i, v in enumerate(corr.values):
        ax.text(v + (0.01 if v > 0 else -0.01), i, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8)
    ax.set_xlim(-0.65, 0.65)
    ax.set_xlabel("Pearson correlation with target")
    ax.set_title("Which features move with heart disease?")
    return fig


def categorical_disease_rate(df: pd.DataFrame, columns: list) -> plt.Figure:
    """Share of patients with disease within each category, with group sizes."""
    pal = _palette()
    labelled = with_labels(df)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    overall = df["target"].mean()
    for ax, col in zip(axes.flat, columns):
        stats = labelled.groupby(col, observed=True)["target"].agg(["mean", "size"]).sort_values("mean")
        colors = [pal[1] if m > overall else pal[0] for m in stats["mean"]]
        ax.barh(stats.index.astype(str), stats["mean"], color=colors)
        for i, (m, n) in enumerate(zip(stats["mean"], stats["size"])):
            ax.text(m + 0.01, i, f"{m:.0%}  (n={n})", va="center", fontsize=8)
        ax.axvline(overall, color="grey", ls="--", lw=1)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Disease rate")
        ax.set_title(PRETTY.get(col, col), fontsize=10)
    fig.suptitle(f"Disease rate by category (dashed line = overall {overall:.0%})",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def age_band_risk(df: pd.DataFrame) -> plt.Figure:
    """Disease rate per age band split by sex: how risk climbs with age."""
    pal = _palette()
    bands = pd.cut(df["age"], bins=[28, 40, 50, 60, 70, 80], right=False,
                   labels=["<40", "40-49", "50-59", "60-69", "70+"])
    table = (df.assign(band=bands, sex=df["sex"].map({0: "Female", 1: "Male"}))
               .groupby(["band", "sex"], observed=False)["target"].agg(["mean", "size"])
               .reset_index())
    labels = list(bands.cat.categories)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sex, marker, color in [("Male", "o", pal["accent"]), ("Female", "s", pal[1])]:
        part = table[table["sex"] == sex]
        xs = range(len(part))
        ax.plot(xs, part["mean"], marker=marker, color=color, lw=2, label=sex)
        for x, y, n in zip(xs, part["mean"], part["size"]):
            if n:
                ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=color)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Age band")
    ax.set_ylabel("Disease rate")
    ax.set_title("Heart disease rate by age band and sex")
    ax.legend()
    return fig


def hr_vs_age(df: pd.DataFrame) -> plt.Figure:
    """Max heart rate against age, with the textbook 220 - age reference line."""
    pal = _palette()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.scatterplot(data=df, x="age", y="thalach", hue="target", palette=pal, alpha=0.75,
                    s=35, edgecolor="white", ax=ax)
    ages = np.linspace(df["age"].min(), df["age"].max(), 50)
    ax.plot(ages, 220 - ages, color="grey", ls="--", lw=1, label="220 - age")
    ax.set_xlabel(PRETTY["age"])
    ax.set_ylabel(PRETTY["thalach"])
    ax.set_title("Patients with disease reach lower peak heart rates")
    ax.legend(title="Target")
    return fig


def generate_all(save: bool = True) -> dict:
    cfg = load_config()
    set_style()
    out_dir = resolve(cfg["eda"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(resolve(cfg["data"]["clean_file"]))
    raw = load_raw(resolve(cfg["data"]["raw_file"]))
    numeric = cfg["columns"]["numeric"]

    figures = {
        "01_class_balance": class_balance(df),
        "02_missing_values": missing_values(raw),
        "03_numeric_histograms": numeric_histograms(df, numeric),
        "04_numeric_boxplots": numeric_boxplots(df, numeric),
        "05_correlation_heatmap": correlation_heatmap(df),
        "06_target_correlation": target_correlation(df),
        "07_categorical_disease_rate": categorical_disease_rate(
            df, ["cp", "thal", "ca", "exang", "slope", "sex"]),
        "08_age_band_risk": age_band_risk(df),
        "09_hr_vs_age": hr_vs_age(df),
    }
    if save:
        for name, fig in figures.items():
            fig.savefig(out_dir / f"{name}.png")
            plt.close(fig)
        log.info("Saved %d figures to %s", len(figures), out_dir)
    return figures


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    generate_all()


if __name__ == "__main__":
    main()
