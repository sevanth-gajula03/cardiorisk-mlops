"""Builds notebooks/02_model_and_inference.ipynb (training results + inference).

Training itself runs as a script (`make train`); this notebook reviews the
results and exercises the packaged model the same way the API does.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("# 02 - Model results and inference\n\n"
       "Training is run with `python -m heartrisk.models.train` (also in CI). This notebook "
       "reviews what that session produced and shows how the packaged model is used for "
       "inference - the same `HeartRiskModel` wrapper the FastAPI service loads."),
    code("import json\n"
         "import pandas as pd\n"
         "from IPython.display import Image, display\n\n"
         "from heartrisk.config import resolve\n"
         "from heartrisk.models.predict import HeartRiskModel\n\n"
         "pd.set_option('display.width', 140)"),

    md("## 1. Model comparison\n\n"
       "Three model families, each tuned with and without the engineered features "
       "(5-fold stratified CV on the training split; test = 20% hold-out)."),
    code("summary = pd.read_csv(resolve('reports/model_comparison.csv'))\n"
         "summary.round(3)"),
    code("display(Image(filename=str(resolve('reports/figures/model/10_model_comparison.png')), width=650))"),
    md("Logistic regression with the engineered features has the best and most stable CV "
       "ROC-AUC and the smallest train-CV gap, so it is the champion. The engineered "
       "features improve CV ROC-AUC for every model family:"),
    code("s = summary.assign(model=summary.run_name.str.split('-').str[0],\n"
         "                   features=summary.run_name.str.split('-').str[1])\n"
         "s.pivot(index='model', columns='features', values='cv_roc_auc')"
         ".assign(gain=lambda d: d['eng'] - d['base']).round(4)"),

    md("## 2. Champion on the hold-out set"),
    code("fig_dir = resolve('reports/figures/model')\n"
         "display(Image(filename=str(fig_dir / '13_champion_roc_pr.png'), width=800))\n"
         "display(Image(filename=str(fig_dir / '12_champion_confusion.png'), width=380))"),

    md("## 3. Experiment tracking\n\n"
       "Runs are logged to the local MLflow store (`mlflow.db`, created by `make train`)."),
    code("import mlflow\n\n"
         "db = resolve('mlflow.db')\n"
         "if db.exists():\n"
         "    mlflow.set_tracking_uri(f'sqlite:///{db}')\n"
         "    runs = mlflow.search_runs(experiment_names=['cardiorisk-heart-disease'],\n"
         "                              filter_string=\"tags.stage = 'tuning'\",\n"
         "                              order_by=['metrics.cv_roc_auc DESC'])\n"
         "    display(runs[['tags.mlflow.runName', 'params.model_type', 'params.engineered_features',\n"
         "                  'metrics.cv_roc_auc', 'metrics.test_roc_auc', 'metrics.test_recall']]\n"
         "            .head(6).round(3))\n"
         "else:\n"
         "    print('No local MLflow store yet - run `make train` first.')"),

    md("## 4. Inference with the packaged model\n\n"
       "`models/` holds the full pipeline (features, preprocessing, model) plus metadata."),
    code("model = HeartRiskModel.load()\n"
         "print('version :', model.version)\n"
         "print('type    :', model.metadata['model_type'])\n"
         "keys = ['cv_roc_auc', 'test_roc_auc', 'test_recall']\n"
         "print('metrics :', {k: model.metadata['metrics'][k] for k in keys})"),
    code("patients = json.loads(resolve('samples/patients.json').read_text())\n"
         "results = model.predict(patients, explain=True)\n"
         "cols = ['prediction', 'probability', 'confidence', 'risk_band']\n"
         "pd.DataFrame(patients).join(pd.DataFrame(results)[cols])"),
    md("Each prediction also explains itself - the three features that moved this patient's "
       "log-odds the most (coefficient x standardised value):"),
    code("for p, r in zip(patients, results, strict=True):\n"
         "    factors = ', '.join(f\"{f['feature']} ({f['weight']:+.2f})\" for f in r['top_factors'])\n"
         "    who = f\"age {p['age']}, sex {p['sex']}\"\n"
         "    print(f\"{who}: {r['label']} p={r['probability']:.2f} -> {factors}\")"),

    md("## 5. Same answer from the MLflow format\n\n"
       "The MLflow copy of the model gives identical predictions, so either format can be served."),
    code("pyfunc = mlflow.pyfunc.load_model(str(resolve('models/mlflow_model')))\n"
         "frame = pd.DataFrame(patients)[model.metadata['features']].astype(float)\n"
         "print('MLflow  :', [int(x) for x in pyfunc.predict(frame)])\n"
         "print('wrapper :', [r['prediction'] for r in results])"),

    md("## 6. Input validation\n\n"
       "Missing fields are rejected rather than silently imputed (the API adds range checks on top)."),
    code("try:\n"
         "    model.predict([{k: v for k, v in patients[0].items() if k != 'thalach'}])\n"
         "except ValueError as exc:\n"
         "    print('rejected:', exc)"),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "02_model_and_inference.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
