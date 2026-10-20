"""Train, tune, compare and package the heart disease classifiers.

One training session = one parent MLflow run. Inside it, every model type is
tuned with a stratified grid search, optionally twice (with and without the
engineered features), and each combination is logged as a nested child run
with its parameters, CV and test metrics, plots and the fitted pipeline.

The configuration with the best mean CV ROC-AUC is chosen, refitted on all
rows, saved to models/ and registered in the MLflow model registry.

Usage:
    python -m heartrisk.models.train
    python -m heartrisk.models.train --models logistic_regression --no-ablation
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import sklearn
import skops.io as sio
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heartrisk import __version__
from heartrisk.config import load_config, resolve
from heartrisk.data.download import sha256_of
from heartrisk.eda.plots import set_style
from heartrisk.features import RAW_FEATURES, build_pipeline
from heartrisk.models import evaluate

log = logging.getLogger(__name__)

SHORT_NAMES = {
    "logistic_regression": "logreg",
    "random_forest": "rf",
    "gradient_boosting": "gbm",
}


def make_estimator(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def param_grid(grid: dict) -> dict:
    return {f"model__{k}": v for k, v in grid.items()}


def trusted_types(model) -> list[str]:
    """Types MLflow's skops serializer must be told to trust (our own transformer, numpy dtypes)."""
    return sio.get_untrusted_types(data=sio.dumps(model))


def setup_mlflow(cfg: dict) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["tracking_uri"])
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{resolve(uri.removeprefix('sqlite:///'))}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(cfg["experiment_name"]) is None:
        mlflow.create_experiment(cfg["experiment_name"],
                                 artifact_location=resolve(cfg["artifact_dir"]).as_uri())
    mlflow.set_experiment(cfg["experiment_name"])
    return uri


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.Series, str]:
    path = resolve(cfg["data"]["clean_file"])
    df = pd.read_csv(path)
    # Features are float64 so the logged model schema also accepts decimals and
    # matches what the API sends (JSON numbers).
    X = df[RAW_FEATURES].astype("float64")
    return X, df[cfg["columns"]["target"]], sha256_of(path.read_bytes())


def log_figure(fig, name: str) -> None:
    mlflow.log_figure(fig, f"plots/{name}.png")
    plt.close(fig)


def run_search(model_name: str, engineered: bool, data: dict, cfg: dict) -> dict:
    """Tune one model / feature-set combination inside a nested MLflow run."""
    seed = cfg["project"]["random_seed"]
    tcfg = cfg["training"]
    run_name = f"{SHORT_NAMES[model_name]}-{'eng' if engineered else 'base'}"
    grid = param_grid(tcfg["models"][model_name])
    cv = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=seed)

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        search = GridSearchCV(
            build_pipeline(make_estimator(model_name, seed), engineered=engineered),
            grid, scoring=evaluate.CV_SCORING, refit=tcfg["selection_metric"],
            cv=cv, n_jobs=-1, return_train_score=True,
        )
        search.fit(data["X_train"], data["y_train"])
        best = search.best_index_
        res = search.cv_results_

        cv_metrics = {}
        for metric in evaluate.CV_SCORING:
            cv_metrics[f"cv_{metric}"] = res[f"mean_test_{metric}"][best]
            cv_metrics[f"cv_{metric}_std"] = res[f"std_test_{metric}"][best]
        cv_metrics["train_roc_auc"] = res["mean_train_roc_auc"][best]
        cv_metrics["overfit_gap"] = cv_metrics["train_roc_auc"] - cv_metrics["cv_roc_auc"]

        y_prob = search.best_estimator_.predict_proba(data["X_test"])[:, 1]
        test_metrics = {f"test_{k}": v for k, v in
                        evaluate.compute_metrics(data["y_test"], y_prob).items()}

        best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        mlflow.set_tags({"model_type": model_name, "stage": "tuning"})
        mlflow.log_params({
            "model_type": model_name,
            "engineered_features": engineered,
            "cv_folds": tcfg["cv_folds"],
            "grid_size": len(res["params"]),
            **{f"best_{k}": v for k, v in best_params.items()},
        })
        mlflow.log_dict({k.removeprefix("model__"): v for k, v in grid.items()}, "search_space.json")
        mlflow.log_metrics({**cv_metrics, **test_metrics})

        title = f"{run_name} (test set)"
        log_figure(evaluate.plot_confusion(data["y_test"], y_prob, title), "confusion_matrix")
        log_figure(evaluate.plot_roc_pr(data["y_test"], y_prob, title), "roc_pr")
        log_figure(evaluate.plot_calibration(data["y_test"], y_prob, f"{run_name} calibration"),
                   "calibration")
        importance = evaluate.feature_importance(search.best_estimator_)
        log_figure(evaluate.plot_importance(importance, f"{run_name} feature importance"),
                   "feature_importance")

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(res).drop(columns="params").to_csv(Path(tmp) / "cv_results.csv", index=False)
            importance.rename("importance").to_csv(Path(tmp) / "feature_importance.csv")
            mlflow.log_artifacts(tmp, artifact_path="tables")

        signature = infer_signature(data["X_train"], search.best_estimator_.predict_proba(data["X_train"]))
        mlflow.sklearn.log_model(search.best_estimator_, name="model", signature=signature,
                                 input_example=data["X_train"].head(3),
                                 skops_trusted_types=trusted_types(search.best_estimator_))

        log.info("%-10s cv_roc_auc=%.4f (+/- %.4f)  test_roc_auc=%.4f  test_recall=%.3f",
                 run_name, cv_metrics["cv_roc_auc"], cv_metrics["cv_roc_auc_std"],
                 test_metrics["test_roc_auc"], test_metrics["test_recall"])

        return {
            "run_name": run_name,
            "run_id": run.info.run_id,
            "model_type": model_name,
            "engineered": engineered,
            "params": best_params,
            "y_prob": y_prob,
            **cv_metrics,
            **test_metrics,
        }


def save_champion(champion: dict, data: dict, cfg: dict, data_sha: str, parent_run_id: str) -> dict:
    """Refit the winning configuration and write the deployable artifacts."""
    seed = cfg["project"]["random_seed"]
    model_dir = resolve(cfg["training"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(make_estimator(champion["model_type"], seed),
                              engineered=champion["engineered"])
    pipeline.set_params(**{f"model__{k}": v for k, v in champion["params"].items()})
    if cfg["training"]["refit_on_full_data"]:
        X_fit, y_fit = data["X_all"], data["y_all"]
    else:
        X_fit, y_fit = data["X_train"], data["y_train"]
    pipeline.fit(X_fit, y_fit)

    trained_at = datetime.now(timezone.utc)
    metadata = {
        "model_name": cfg["mlflow"]["registered_model_name"],
        "model_version": trained_at.strftime("%Y.%m.%d.%H%M"),
        "package_version": __version__,
        "model_type": champion["model_type"],
        "engineered_features": champion["engineered"],
        "params": champion["params"],
        "decision_threshold": 0.5,
        "features": RAW_FEATURES,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "trained_on_rows": len(X_fit),
        "data_sha256": data_sha,
        "sklearn_version": sklearn.__version__,
        "mlflow_parent_run_id": parent_run_id,
        "mlflow_tuning_run_id": champion["run_id"],
        "metrics": {k: round(float(v), 4) for k, v in champion.items()
                    if k.startswith(("cv_", "test_")) and not k.endswith("_std")},
    }

    joblib.dump(pipeline, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    mlflow_dir = model_dir / "mlflow_model"
    shutil.rmtree(mlflow_dir, ignore_errors=True)
    signature = infer_signature(data["X_all"], pipeline.predict_proba(data["X_all"]))
    mlflow.sklearn.save_model(pipeline, mlflow_dir, signature=signature,
                              input_example=data["X_all"].head(3),
                              skops_trusted_types=trusted_types(pipeline))

    # Register the shipped model and point the "champion" alias at it.
    info = mlflow.sklearn.log_model(pipeline, name="champion_model", signature=signature,
                                    input_example=data["X_all"].head(3),
                                    skops_trusted_types=trusted_types(pipeline),
                                    registered_model_name=cfg["mlflow"]["registered_model_name"])
    client = MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(cfg["mlflow"]["registered_model_name"], "champion", version)
    client.set_model_version_tag(cfg["mlflow"]["registered_model_name"], version,
                                 "model_type", champion["model_type"])
    mlflow.log_dict(metadata, "champion_metadata.json")
    metadata["registry_version"] = version
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def write_reports(summary: pd.DataFrame, champion: dict, data: dict, metadata: dict) -> None:
    """Save comparison figures and a markdown summary for the written report."""
    fig_dir = resolve("reports/figures/model")
    fig_dir.mkdir(parents=True, exist_ok=True)
    title = f"Champion: {champion['run_name']}"
    figures = {
        "10_model_comparison": evaluate.plot_model_comparison(summary),
        "11_roc_overlay": evaluate.plot_roc_overlay(
            {r["run_name"]: r["y_prob"] for r in data["results"]}, data["y_test"]),
        "12_champion_confusion": evaluate.plot_confusion(data["y_test"], champion["y_prob"], title),
        "13_champion_roc_pr": evaluate.plot_roc_pr(data["y_test"], champion["y_prob"], title),
        "14_champion_calibration": evaluate.plot_calibration(
            data["y_test"], champion["y_prob"], "Champion calibration"),
    }
    for name, fig in figures.items():
        fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        mlflow.log_figure(fig, f"comparison/{name}.png")
        plt.close(fig)

    cols = ["run_name", "cv_roc_auc", "cv_roc_auc_std", "cv_accuracy", "cv_precision", "cv_recall",
            "cv_f1", "test_roc_auc", "test_accuracy", "test_precision", "test_recall", "overfit_gap"]
    table = summary[cols].sort_values("cv_roc_auc", ascending=False)
    lines = [
        "# Model comparison",
        "",
        f"Generated {metadata['trained_at']} - champion **{champion['run_name']}** "
        f"(registry version {metadata['registry_version']}).",
        "",
        "Selection rule: highest mean 5-fold CV ROC-AUC on the training split. "
        "Test metrics come from a stratified 20% hold-out that was never used for tuning.",
        "",
        "| Run | CV ROC-AUC | CV acc | CV prec | CV recall | CV F1 | Test ROC-AUC | Test acc "
        "| Test prec | Test recall | Overfit gap |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r.run_name} | {r.cv_roc_auc:.3f} +/- {r.cv_roc_auc_std:.3f} | {r.cv_accuracy:.3f} "
            f"| {r.cv_precision:.3f} | {r.cv_recall:.3f} | {r.cv_f1:.3f} | {r.test_roc_auc:.3f} "
            f"| {r.test_accuracy:.3f} | {r.test_precision:.3f} | {r.test_recall:.3f} "
            f"| {r.overfit_gap:.3f} |")
    lines += ["", "## Champion hyper-parameters", "", "```json",
              json.dumps(champion["params"], indent=2), "```", ""]
    resolve("reports/model_comparison.md").write_text("\n".join(lines))
    table.to_csv(resolve("reports/model_comparison.csv"), index=False)
    mlflow.log_artifact(str(resolve("reports/model_comparison.md")))


def train(model_names: list | None = None, ablation: bool | None = None) -> dict:
    cfg = load_config()
    tcfg = cfg["training"]
    seed = cfg["project"]["random_seed"]
    model_names = model_names or list(tcfg["models"])
    ablation = tcfg["feature_ablation"] if ablation is None else ablation
    feature_sets = [False, True] if ablation else [True]

    set_style()
    tracking_uri = setup_mlflow(cfg["mlflow"])
    log.info("MLflow tracking URI: %s", tracking_uri)

    X, y, data_sha = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tcfg["test_size"], stratify=y, random_state=seed)
    data = {"X_all": X, "y_all": y, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}
    log.info("Train rows: %d, test rows: %d", len(X_train), len(X_test))

    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=f"session-{session}") as parent:
        mlflow.set_tag("stage", "session")
        mlflow.log_params({
            "models": ",".join(model_names),
            "feature_ablation": ablation,
            "test_size": tcfg["test_size"],
            "cv_folds": tcfg["cv_folds"],
            "selection_metric": tcfg["selection_metric"],
            "random_seed": seed,
            "data_sha256": data_sha,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })
        mlflow.log_input(
            mlflow.data.from_pandas(pd.concat([X, y], axis=1), name="heart_clean",
                                    targets=cfg["columns"]["target"],
                                    source=str(resolve(cfg["data"]["clean_file"]))),
            context="training")
        mlflow.log_artifact(str(resolve("config/params.yaml")))

        results = [run_search(m, eng, data, cfg) for m in model_names for eng in feature_sets]
        data["results"] = results

        summary = pd.DataFrame([{k: v for k, v in r.items() if k != "y_prob"} for r in results])
        metric = f"cv_{tcfg['selection_metric']}"
        champion = max(results, key=lambda r: r[metric])
        log.info("Champion: %s (%s = %.4f)", champion["run_name"], metric, champion[metric])

        mlflow.set_tags({"champion": champion["run_name"], "champion_run_id": champion["run_id"]})
        mlflow.log_metrics({f"champion_{k}": v for k, v in champion.items()
                            if k.startswith(("cv_", "test_"))})

        metadata = save_champion(champion, data, cfg, data_sha, parent.info.run_id)
        write_reports(summary, champion, data, metadata)

    log.info("Saved model %s to %s", metadata["model_version"], resolve(tcfg["model_dir"]))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare heart disease models")
    parser.add_argument("--models", nargs="+", choices=list(SHORT_NAMES),
                        help="subset of models to train (default: all in params.yaml)")
    parser.add_argument("--no-ablation", action="store_true",
                        help="only train with engineered features")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    train(args.models, ablation=False if args.no_ablation else None)


if __name__ == "__main__":
    main()
