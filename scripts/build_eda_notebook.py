"""Builds notebooks/01_eda.ipynb from the cells below.

Keeping the notebook source in a script makes diffs readable; run
`make eda` (or this script followed by nbconvert --execute) to refresh it.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("# 01 - Data Acquisition & Exploratory Data Analysis\n\n"
       "**Project:** CardioRisk - heart disease risk prediction  \n"
       "**Dataset:** UCI Heart Disease, Cleveland subset (303 patients, 13 features + diagnosis)\n\n"
       "This notebook walks through how the data is obtained, cleaned and explored. "
       "All the heavy lifting lives in the `heartrisk` package so the same code runs in "
       "CI and in the training pipeline; the notebook only calls it and explains the results."),
    code("import json\n"
         "import pandas as pd\n"
         "from IPython.display import display\n\n"
         "from heartrisk.config import load_config, resolve\n"
         "from heartrisk.data import download, clean\n"
         "from heartrisk.eda import plots\n\n"
         "cfg = load_config()\n"
         "plots.set_style()\n"
         "pd.set_option('display.width', 120)"),

    md("## 1. Data acquisition\n\n"
       "`heartrisk.data.download` pulls the official UCI zip, extracts "
       "`processed.cleveland.data` and writes a `MANIFEST.json` with the source URL, "
       "download time and SHA-256. If the zip endpoint is down it falls back to the direct "
       "file link. The checksum lets anyone confirm they trained on exactly the same bytes."),
    code("manifest = download.download()\n"
         "print(json.dumps(manifest, indent=2))"),
    code("raw = clean.load_raw(resolve(cfg['data']['raw_file']))\n"
         "print(raw.shape)\n"
         "raw.head()"),

    md("### Feature dictionary\n\n"
       "| Column | Meaning | Type |\n|---|---|---|\n"
       "| age | Age in years | continuous |\n"
       "| sex | 1 = male, 0 = female | binary |\n"
       "| cp | Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic) | nominal |\n"
       "| trestbps | Resting blood pressure (mm Hg) | continuous |\n"
       "| chol | Serum cholesterol (mg/dl) | continuous |\n"
       "| fbs | Fasting blood sugar > 120 mg/dl | binary |\n"
       "| restecg | Resting ECG (0 normal, 1 ST-T abnormality, 2 LV hypertrophy) | nominal |\n"
       "| thalach | Maximum heart rate achieved | continuous |\n"
       "| exang | Exercise-induced angina | binary |\n"
       "| oldpeak | ST depression induced by exercise | continuous |\n"
       "| slope | Slope of peak exercise ST segment | nominal |\n"
       "| ca | Number of major vessels coloured by fluoroscopy (0-3) | ordinal |\n"
       "| thal | Thallium test (3 normal, 6 fixed defect, 7 reversible defect) | nominal |\n"
       "| num | Diagnosis, 0 = no disease, 1-4 = increasing severity | target |"),

    md("## 2. Data quality check"),
    code("raw.info()"),
    code("raw.describe().T.round(2)"),
    code("plots.missing_values(raw);"),
    md("Only **`ca` (4 rows)** and **`thal` (2 rows)** are missing, encoded as `?` in the raw "
       "file. That is under 2% of rows, so instead of dropping patients I impute with the "
       "**mode** - both are coded categorical columns, so a median could produce a "
       "value that is not a valid code."),
    code("raw['num'].value_counts().sort_index()"),
    md("The original diagnosis ranges 0-4, but the higher grades have very few patients "
       "(13 with grade 4). Following the problem statement, I collapse it into a binary "
       "target: **0 = no disease, 1 = any disease (num >= 1)**."),

    md("## 3. Cleaning\n\n"
       "`heartrisk.data.clean` performs these steps and logs each one to "
       "`reports/data_quality.json`:\n\n"
       "1. Parse `?` as missing.\n"
       "2. Drop exact duplicate rows.\n"
       "3. Flag values outside clinically plausible ranges (set in `config/params.yaml`) as missing.\n"
       "4. Impute - median for continuous columns, mode for coded columns.\n"
       "5. Binarise the target and cast coded columns to integers.\n"
       "6. Write `heart_clean.csv` (integer codes) and `heart_encoded.csv` (one-hot nominal columns)."),
    code("df = clean.run()\n"
         "print(json.dumps(json.loads(resolve(cfg['data']['quality_report']).read_text()), indent=2))"),
    code("df.head()"),
    code("encoded = pd.read_csv(resolve(cfg['data']['encoded_file']))\n"
         "print(encoded.shape)\n"
         "encoded.columns.tolist()"),
    md("The encoded file one-hot encodes `cp`, `restecg`, `slope` and `thal` with readable "
       "column names (e.g. `cp_asymptomatic`). `ca` stays as an ordinal integer because "
       "more blocked vessels genuinely means more risk. Scaling is **not** done here - it "
       "belongs inside the model pipeline so it is fitted only on training folds and never "
       "leaks information from the test set."),

    md("## 4. Exploratory analysis\n\n### 4.1 Class balance"),
    code("plots.class_balance(df);"),
    md("164 healthy vs 139 diseased patients (54% / 46%). The classes are close enough "
       "that accuracy is meaningful, but I will still report precision, recall and ROC-AUC, "
       "and use stratified folds so each split keeps this ratio."),

    md("### 4.2 Continuous features"),
    code("plots.numeric_histograms(df, cfg['columns']['numeric']);"),
    code("plots.numeric_boxplots(df, cfg['columns']['numeric']);"),
    code("df.groupby('target')[cfg['columns']['numeric']].median().T.rename(columns={0: 'no disease', 1: 'disease'})"),
    md("**Observations**\n\n"
       "- **thalach** (max heart rate) is the clearest separator: diseased patients peak around "
       "140 bpm vs ~160 bpm for healthy ones.\n"
       "- **oldpeak** is heavily right-skewed; most healthy patients sit at 0, while diseased "
       "patients spread out to 4+. A log or power transform may help linear models.\n"
       "- **age** shifts right for the disease group (median 58 vs 52).\n"
       "- **trestbps** and **chol** overlap heavily - weak signals on their own. "
       "Cholesterol has one extreme reading (564 mg/dl); it is high but clinically possible, "
       "so I keep it and rely on robust models / scaling rather than deleting it."),

    md("### 4.3 Correlations"),
    code("plots.correlation_heatmap(df);"),
    code("plots.target_correlation(df);"),
    md("- Strongest links with the target: **thal (0.52), ca (0.46), exang (0.43), oldpeak (0.42), "
       "cp (0.41)** and **thalach (-0.42)**.\n"
       "- `fbs` (0.03) and `chol` (0.09) carry almost no linear signal.\n"
       "- No pair of features is above 0.6, so multicollinearity is not a serious concern; "
       "the largest is `oldpeak`-`slope` (0.58), which makes sense as both describe the ST segment.\n"
       "- Pearson correlation on nominal codes (`cp`, `thal`) is only indicative - the "
       "category-level view below is more honest for them."),

    md("### 4.4 Categorical features"),
    code("plots.categorical_disease_rate(df, ['cp', 'thal', 'ca', 'exang', 'slope', 'sex']);"),
    md("- **Asymptomatic chest pain** has a 73% disease rate - counter-intuitive but well known "
       "in this dataset: patients without classic angina symptoms were often referred for "
       "silent ischaemia.\n"
       "- A **reversible thallium defect** (76%) and **any coloured vessel** (68-85%) are strong red flags.\n"
       "- **Exercise-induced angina** more than doubles the rate (77% vs 31%).\n"
       "- These relationships are not monotonic in the raw codes (e.g. `cp` 1-3 are all low), "
       "which is why the nominal columns are one-hot encoded rather than treated as numbers."),

    md("### 4.5 Age, sex and heart rate"),
    code("plots.age_band_risk(df);"),
    md("Risk climbs steadily with age for men, from ~40% under 50 to ~75% in their 60s. "
       "Women in this sample have much lower rates at every age band, although the female "
       "group is small (97 patients) so the 70+ point (n=5) should not be over-read. "
       "This suggests an **age x sex interaction** worth testing during feature engineering."),
    code("plots.hr_vs_age(df);"),
    md("Max heart rate falls with age for everyone, but diseased patients sit consistently "
       "further below the `220 - age` reference. That gap - **heart-rate reserve** - could "
       "be a more informative feature than `thalach` alone, because it adjusts for age."),

    md("## 5. Summary and next steps\n\n"
       "| Finding | Implication for modelling |\n|---|---|\n"
       "| Only 6 missing cells, all in coded columns | Mode imputation inside the pipeline |\n"
       "| Classes are 54/46 | Stratified CV; report ROC-AUC, precision, recall |\n"
       "| `oldpeak` right-skewed | Try a power transform for linear models |\n"
       "| Nominal codes are non-monotonic | One-hot encode `cp`, `restecg`, `slope`, `thal` |\n"
       "| `fbs`, `chol` weak | Keep, but check feature importance |\n"
       "| Heart rate depends on age | Engineer `hr_reserve = thalach / (220 - age)` |\n"
       "| Risk differs strongly by sex across age | Consider an age-sex interaction |\n\n"
       "All figures are saved to `reports/figures/` for the report."),
    code("plots.generate_all()\n"
         "sorted(p.name for p in resolve(cfg['eda']['figures_dir']).glob('*.png'))"),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "01_eda.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
