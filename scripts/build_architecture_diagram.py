"""Draws the end-to-end architecture diagram used in the README and report.

    python scripts/build_architecture_diagram.py   # -> reports/figures/architecture.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "architecture.png"

TEAL, CORAL, NAVY, SAND, GREY = "#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#6c757d"
LIGHT = {TEAL: "#e6f4f2", CORAL: "#fcebe6", NAVY: "#e5eaec", SAND: "#fbf4e1", GREY: "#f1f3f5"}

fig, ax = plt.subplots(figsize=(15, 9.2))
ax.set_xlim(0, 150)
ax.set_ylim(0, 92)
ax.axis("off")


def zone(x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                                fc=LIGHT[color], ec=color, lw=1.6, ls="--"))
    ax.text(x + 1.5, y + h - 2.2, title, fontsize=11, fontweight="bold", color=color, va="top")


def box(x, y, w, h, title, lines=(), color=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc="white", ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=color)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 4.6 - i * 2.4, line, ha="center", va="top", fontsize=7.8,
                color="#333333")


def arrow(x1, y1, x2, y2, label="", color=GREY, rad=0.0, label_dy=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.4, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label, ha="center", fontsize=7.5,
                color=color, style="italic")


ax.text(75, 90.5, "CardioRisk - end-to-end MLOps architecture", ha="center", fontsize=15,
        fontweight="bold", color=NAVY)

# --- Development / training -------------------------------------------------
zone(1, 48, 58, 38, "1  Data & training (local / CI)", TEAL)
box(3, 67, 16, 11, "UCI repository", ["Cleveland heart", "disease data"], TEAL)
box(21.5, 67, 18, 11, "download + clean", ["sha256 manifest", "range checks, impute"], TEAL)
box(42, 67, 15, 11, "EDA", ["9 figures", "executed notebook"], TEAL)
box(3, 51, 25, 13, "train.py", ["GridSearchCV x 3 models", "x with/without features",
                                  "stratified 5-fold CV"], TEAL)
box(32, 51, 25, 13, "MLflow", ["params, metrics, plots", "models, dataset",
                               "registry @champion"], TEAL)
arrow(19, 72.5, 21.5, 72.5)
arrow(39.5, 72.5, 42, 72.5)
arrow(30.5, 67, 15, 64, rad=0.15)
arrow(28, 57.5, 32, 57.5, "log")

# --- Packaging ----------------------------------------------------------------
zone(62, 48, 34, 38, "2  Packaging", SAND)
box(64, 67, 30, 11, "models/", ["model.joblib (full pipeline)", "mlflow_model/, metadata.json"],
    "#b08900")
box(64, 51, 30, 13, "Docker image", ["FastAPI + uvicorn", "non-root, healthcheck",
                                     "tag = model version"], "#b08900")
arrow(57, 61, 64, 72.5, "champion", rad=-0.2)
arrow(79, 67, 79, 64)

# --- CI/CD -------------------------------------------------------------------------
zone(99, 48, 50, 38, "3  CI/CD - GitHub Actions", CORAL)
steps = [("lint", "ruff"), ("test", "60 tests, cov"), ("train", "quality gate"),
         ("docker", "smoke test")]
for i, (name, sub) in enumerate(steps):
    box(101 + i * 12, 67, 10.5, 11, name, [sub], CORAL)
    if i:
        arrow(101 + i * 12 - 1.5, 72.5, 101 + i * 12, 72.5, color=CORAL)
box(101, 51, 22, 11, "manifests", ["kustomize +", "kubeconform"], CORAL)
box(126, 51, 21, 11, "GHCR", ["cardiorisk-api:", "<model-version>"], CORAL)
arrow(142, 67, 137, 62, "push (main)", color=CORAL)
ax.text(124, 45.8, "artifacts: test-reports, model, reports, mlflow-runs, container-logs",
        ha="center", fontsize=7.5, color=CORAL)

# --- Kubernetes ------------------------------------------------------------------
zone(1, 2, 100, 43, "4  Minikube cluster", NAVY)
box(4, 28, 17, 11, "Client", ["curl / Swagger /", "load_test.py"], GREY)
box(25, 28, 19, 11, "Ingress (nginx)", ["cardiorisk.local", "rate + size limits"], NAVY)
box(25, 12, 19, 11, "Service", ["LoadBalancer :8090", "(minikube tunnel)"], NAVY)
ax.add_patch(FancyBboxPatch((49, 6), 30, 34, boxstyle="round,pad=0.3,rounding_size=1.5",
                            fc="white", ec=NAVY, lw=1.2, ls=":"))
ax.text(64, 38.3, "namespace cardiorisk", ha="center", fontsize=8.5, color=NAVY, fontweight="bold")
for i in range(3):
    box(52, 27 - i * 8.3, 24, 6.8, f"api pod {i + 1}" + (" (HPA)" if i == 2 else ""),
        ["/predict  /health  /metrics"], NAVY)
ax.text(64, 7.3, "HPA 2-5 pods @70% CPU, PDB, probes", ha="center", fontsize=7.5, color=NAVY)
arrow(21, 33.5, 25, 33.5)
arrow(21, 31, 25, 18, rad=-0.2)
arrow(44, 33.5, 52, 30)
arrow(44, 17.5, 52, 22)

# --- Monitoring --------------------------------------------------------------------
box(84, 26, 15, 13, "Prometheus", ["pod discovery", "cAdvisor", "alert rules"], CORAL)
box(84, 8, 15, 13, "Grafana", ["provisioned", "dashboard", "15 panels"], CORAL)
arrow(84, 32, 76, 24, "scrape", color=CORAL, rad=0.1)
arrow(91.5, 26, 91.5, 21, color=CORAL)
ax.text(91.5, 4.0, "namespace monitoring", ha="center", fontsize=7.5, color=CORAL)

# --- Deploy path ------------------------------------------------------------------
zone(104, 2, 45, 43, "5  Release flow", GREY)
flow = ["make deploy", "build image (model tag)", "minikube image load",
        "kubectl apply -k (kustomize)", "rolling update, maxUnavailable 0", "smoke test via ingress"]
for i, step in enumerate(flow):
    y = 37 - i * 5.3
    ax.text(108, y, f"{i + 1}. {step}", fontsize=8.5, color="#333333", va="center")
ax.text(108, 5.5, "rollback: kubectl rollout undo", fontsize=8, color=GREY, style="italic")

arrow(79, 51, 64, 40, "image", color="#b08900", rad=0.25)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
