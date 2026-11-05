# AgroSynapse

> **Agricultural data analysis in one place**: check a dataset's quality, compare crops and
> places, follow trends, and forecast next season's yield.

---

## Overview

**AgroSynapse** is a Streamlit dashboard and Python pipeline for the work an analyst does with a
new agricultural dataset, such as a field-survey export or published crop statistics:

1. **Check it.** The *Data Quality* page flags blank cells, duplicate records, impossible values
   (negative areas, percentages over 100), numbers stored as text, inconsistently spelled labels
   and extreme values, and scores the file out of 100.
2. **Describe it.** The *Crop Insights* page compares groups with 95% confidence intervals, fits
   trends over time, benchmarks one country or region against its peers, and writes the findings
   out in plain sentences. A toggle narrows the view to IITA's mandate crops (cassava, yam, maize,
   cowpea, soybean, banana and plantain).
3. **Model it.** A leakage-aware pipeline trains a yield model, scores it on years it never saw,
   and compares it with the forecast a practitioner would make without a model.

Both analysis pages work on any CSV a visitor uploads, not only the demo data.

The repository ships a runnable demo built from public data: a **25,305-row crop-yield panel
covering 54 African countries, 23 crops and 1990-2024**, assembled from FAOSTAT and World Bank
sources by `scripts/build_dataset.py`. Each row is a **national** figure for one country, crop
and year. Forecasting next season's yield on 2018-2024 records the model never saw, it reaches
**R2 0.977 with a mean absolute error of 470 kg/ha**, and it beats a carry-forward benchmark by a
third. See [Model Quality](#model-quality).

### Example findings from the demo data

Computed by the Crop Insights page, with Nigeria as the focus country and IITA crops selected:

- Over 2020-2024, Nigeria's cassava yield averaged **5,952 kg/ha, 36% below** the median of the
  other African producers (9,316 kg/ha).
- On the same basis Nigeria is **ahead** on cowpea (+74%), maize (+32%) and yam (+13%).
- Across 1990-2024, Nigeria's cassava (-1.5% a year) and yam (-1.0% a year) yield trends
  **fall**, while soybean, cowpea and maize trends **rise** (all p < 0.01).

These are national statistics, so they describe countries, not individual farms. A falling
national trend can also reflect FAOSTAT revisions or expansion onto poorer land, which this
data cannot separate.

### Scope and limits

- **Unit of analysis.** The demo data is national, so it says nothing about individual farms.
  Farm-level work needs farm-level data, which the upload path and cleaning pipeline accept.
- **Inputs.** The demo features are yield history, area harvested and country-level indicators
  (fertilizer use, land use, rural population, GDP, electrification, long-run average
  precipitation). There is no soil, seasonal weather or management-practice data in it.
- **Statistics.** Rows for the same place repeat across years, so they are not independent. The
  confidence intervals and Kruskal-Wallis tests describe the data as given and are a first look,
  not a substitute for a mixed model.
- **KoboToolbox.** `scripts/ingest_kobo.py` and `scripts/clean_data.py` read KoboToolbox CSV
  exports (delimiter detection, metadata columns dropped). The demo itself uses public
  statistics, not Kobo data.

---

## Technology Stack

```text
CSV input: public statistics (FAOSTAT + World Bank), KoboToolbox exports, or any upload
  -> Python ETL (pandas)
  -> Descriptive analysis (pandas / SciPy): quality checks, group comparisons, trends
  -> ML models (scikit-learn Random Forest, Extra Trees) with leakage-aware feature selection
  -> Streamlit dashboard + Altair charts
  -> FastAPI inference API
```

---

## Key Features

- **Data quality report** for any CSV: 13 checks, a severity-ranked findings table with
  examples, a per-column profile, and a CSV download of the findings
- **Crop Insights**: group comparison with confidence intervals and a rank-based test, trend
  lines (with an index-to-100 view for crops at very different yield levels), peer
  benchmarking, and a plain-language summary built only from computed numbers
- **Reproducible data**: `scripts/build_dataset.py` rebuilds the demo panel from FAOSTAT and
  World Bank sources, so every committed row is traceable
- **Cleaning pipeline**: column-name normalisation, KoboToolbox metadata removal, and
  imputation and one-hot encoding inside the model pipeline
- **Leakage-aware feature selection**: identifiers and target-derived columns are blocked, and
  an allowlist that does not fit the data is detected rather than silently applied
- **Honest evaluation**: a forward-looking temporal holdout scored against a carry-forward
  benchmark, not only a median baseline
- **Fast inference**: sub-second predictions from a ~3 MB model file
- **Safe to host**: read-only demo mode by default; uploads stay in the visitor's session

---

## Project Structure

```text
agrosynapse/
|-- data/
|   |-- raw/
|   |   `-- africa_crop_yield.csv   # demo dataset (FAOSTAT + World Bank)
|   `-- processed/                  # clean_data.csv, features.csv,
|                                   # model_pipeline.joblib, model_metrics.json
|-- scripts/
|   |-- analysis.py                 # quality checks, group comparisons, trends, summaries
|   |-- build_dataset.py            # rebuilds the demo dataset from public sources
|   |-- ingest_kobo.py
|   |-- clean_data.py
|   |-- feature_config.py           # feature allowlist, leakage + risky-column policy
|   |-- transform_utils.py
|   |-- transform_features.py
|   |-- train_model.py
|   |-- predict.py
|   `-- run_pipeline.py
|-- tests/                          # pytest suite
|-- streamlit_app.py                # Analytics dashboard
|-- app_api.py                      # FastAPI inference service
|-- assets/                         # Logo and dashboard CSS
|-- .streamlit/config.toml          # Dashboard theme
|-- .python-version                 # 3.11, read by Streamlit Community Cloud
|-- notebooks/
|   |-- 01_eda.ipynb
|   `-- 02_yield_prediction.ipynb
|-- docs/
|   |-- PRD.md
|   |-- UI_SPEC.md
|   `-- DATA_DICTIONARY.md
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```

---

## Quick Start

### Prerequisites
- **Python 3.11** (pinned in `.python-version`). The committed `model_pipeline.joblib` is a
  pickle of a scikit-learn 1.9.0 estimator, and `requirements.txt` pins the exact versions it
  was trained and verified against. Loading it under a different scikit-learn minor version
  raises `InconsistentVersionWarning` and is not guaranteed to predict identically.
- pip

### Installation

```bash
# Clone and setup
git clone https://github.com/vojmax-tech/AgroSynapse.git
cd AgroSynapse

# Create virtual environment
python -m venv .venv

# Activate it -- Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Activate it -- macOS / Linux:
source .venv/bin/activate

# Install dependencies (dashboard + API included)
pip install -r requirements.txt
```

> **Windows:** `source .venv/bin/activate` does not work in PowerShell, and
> pasting a trailing `# On Windows: ...` comment as a command creates a stray
> directory. Use the PowerShell activation line above.

### Run the Dashboard

The repository ships with a trained model and processed data, so the dashboard runs
immediately after install - no pipeline run required:

```bash
# Windows (PowerShell)
$env:AGROSYNAPSE_MODE = "full"; streamlit run streamlit_app.py

# macOS / Linux
AGROSYNAPSE_MODE=full streamlit run streamlit_app.py
```

Open **http://localhost:8501**. Eight pages are available from the sidebar:

| Page | What it shows |
|------|---------------|
| Dashboard | Headline figures, distributions and category breakdowns for the active dataset |
| Crop Insights | Group comparisons, trends, peer benchmarks and a written summary |
| Data Quality | Quality score, findings table and column profile |
| Model Performance | Forward-looking evaluation of the yield model against its benchmarks |
| Data Explorer | Raw, cleaned and feature tables |
| Uploads | Upload a CSV to explore it; run the pipeline (full mode only) |
| Settings | File paths and display options |
| Overview | What the app does and what the demo data is |

`AGROSYNAPSE_MODE=full` enables the write actions (uploads, pipeline runs, saving settings).
Without it the app starts in read-only demo mode, which is the safe default for a shared
deployment - see [Demo mode](#demo-mode-on-by-default).

### Run the Inference API

```bash
uvicorn app_api:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive OpenAPI console.

```bash
# Health check
curl http://localhost:8000/health

# Prediction (one object per record under "rows")
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows":[{"data":{
        "country":"Nigeria","crop":"Maize (corn)","year":2024,
        "area_harvested_ha":5500000,"fertilizer_kg_per_ha":4.2,
        "arable_land_pct":40.5,"agri_land_pct":76.2,"rural_population_pct":37.0,
        "agri_employment_pct":34.1,"gdp_per_capita_usd":2324.6,
        "rural_electricity_pct":23.5,"avg_precipitation_mm":1150,
        "yield_prev_year_kg_per_ha":1939.1,"yield_prev2_year_kg_per_ha":2232.6,
        "yield_prev3_mean_kg_per_ha":2075.2,"area_prev_year_ha":5700000
      }}]}'
```

`/predict` requires every column the loaded pipeline was trained on; missing columns return
a `400` listing them. Retrain the model and the required payload changes with it.

---

## ETL Usage

```bash
# 0) (Optional) Rebuild the demo dataset from FAOSTAT + World Bank. Needs network access
#    and takes about two minutes. The output is committed, so this is only for refreshing.
python scripts/build_dataset.py --output data/raw/africa_crop_yield.csv

# 1) Ingest
python scripts/ingest_kobo.py --input data/raw/africa_crop_yield.csv
# Default output is data/raw/<input_stem>_raw.csv

# Optional: append timestamp for versioned ingest snapshots
python scripts/ingest_kobo.py --input data/raw/africa_crop_yield.csv --timestamp-output

# 2) Clean data
python scripts/clean_data.py --input data/raw/africa_crop_yield_raw.csv --output data/processed/clean_data.csv

# 3) Train model
python scripts/train_model.py --input data/processed/clean_data.csv --target yield_kg_per_ha --output data/processed/model_pipeline.joblib

# 3.5) Generate engineered features for explorer / model diagnostics
python scripts/transform_features.py --input data/processed/clean_data.csv --target yield_kg_per_ha --output-dir data/processed

# 4) Predict
python scripts/predict.py --model data/processed/model_pipeline.joblib --input data/processed/clean_data.csv --output data/processed/predictions.csv --target yield_kg_per_ha

# Or run the full pipeline - this is what regenerates every committed artefact
python scripts/run_pipeline.py --input data/raw/africa_crop_yield.csv --target yield_kg_per_ha --n-jobs -1

# Run the full pipeline on your own raw file
python scripts/run_pipeline.py --input data/raw/your_survey_export.csv --target yield_tons
```


---

## Data Processing Rules

- Normalize column names (lowercase with underscores)
- Drop Kobo metadata fields
- Model pipeline handles imputation and encoding during training/inference

### Feature policy

`scripts/feature_config.py` decides which columns reach the model, under three rules:

1. **Risky columns** are always dropped - identifiers, names, phone numbers and raw dates
   (`RISKY_COLUMN_PATTERNS`). They carry personal data or row-order artefacts, not agronomy.
2. **Leakage columns** are dropped from the feature set but kept in the dataset for
   reporting (`LEAKAGE_COLUMN_PATTERNS`). In the demo panel that is `production_tonnes`,
   which equals `yield_kg_per_ha x area_harvested_ha / 1000` - hand a model both it and
   area harvested and it recovers the target arithmetically.
3. **The allowlist** (`APPROVED_FEATURES`) is a default, not a hard gate. If it covers less
   than half the eligible columns, `resolve_feature_columns()` treats it as written for a
   different schema, falls back to every eligible column, and records that decision in the
   logs and in `model_metrics.json` under `feature_selection`.

Rule 3 exists because of a real failure mode. An allowlist written for one schema used to be
intersected with a different dataset, and whichever one or two columns happened to overlap
became the entire feature set. The pipeline reported success; the only evidence was a
negative R2 buried in the metrics file. `tests/test_feature_config.py` covers that regression.

Check `features_used` in `data/processed/model_metrics.json` after any run on new data.
To train on an explicit set of columns instead:

```bash
python scripts/train_model.py --input data/processed/clean_data.csv --target yield_kg_per_ha   --features "country,crop,year,area_harvested_ha,fertilizer_kg_per_ha,yield_prev_year_kg_per_ha"
```

Pass `--strict-schema` to `run_pipeline.py` to turn an allowlist mismatch into a hard failure
instead of a fallback.

---

## Model Quality

The demo model predicts **`yield_kg_per_ha`** for a country / crop / season from 16 features:
geography, crop, year, area harvested, World Bank agronomic and economic context, and the same
series' yields in previous seasons.

### Forward-looking evaluation

The headline numbers below come from training on **1990-2017** and scoring on **2018-2024**
(5,350 records the model never saw). A random train/test split is not a fair test on a panel
like this - 2019 and 2020 rows for the same country and crop land on opposite sides of it - so
the temporal split is the one to read.

Two baselines are reported, because only one of them is honest:

| Approach | MAE (kg/ha) | RMSE | R2 |
|----------|------------:|-----:|---:|
| **Extra Trees (this model)** | **470** | **1 435** | **0.977** |
| Carry previous season forward | 705 | 2 095 | 0.951 |
| Median of the training data | 5 120 | 10 482 | -0.220 |

Beating the median baseline is meaningless here: every country/crop series has its own level,
so any model that learns those levels scores well. The benchmark that matters is carrying last
season's yield forward, which is what a practitioner would actually do. **The model reduces mean
absolute error against that benchmark by 33%**, so it is adding forecasting value rather than
restating history.

For reference, the random-split figures are MAE 513 / R2 0.965, and 5-fold cross-validation
gives R2 0.959 +/- 0.005.

### What this does and does not claim

- This is **one-season-ahead** forecasting with observed history. Predicting 2020 uses the
  actual 2019 yield, which an extension service would have. It is *not* a multi-year forecast
  made once from 2017.
- The unit of analysis is a **country x crop x year aggregate**, not an individual farm. The
  model captures national-level dynamics; farm-level variance is not represented in the data.
- `production_tonnes` is excluded from the features because it is arithmetically derived from
  the target. Including it would produce a near-perfect score that means nothing.
- Rebuild and re-verify everything with:
  `python scripts/build_dataset.py && python scripts/run_pipeline.py --input data/raw/africa_crop_yield.csv --target yield_kg_per_ha --n-jobs -1`

The dashboard reports all of this in the **Model Performance** page, and `model_metrics.json`
carries the full breakdown, including the cross-validation spread, the candidate model
comparison, and the `feature_selection` decision.

---

## Deployment

### Streamlit Community Cloud

1. Push this repo to GitHub (public, or private on a plan that allows it).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app** -> pick the repo and branch, set **Main file path** to `streamlit_app.py`.
4. Open **Advanced settings** and choose **Python 3.11** - the committed model was built on
   it (see `.python-version`). Optionally paste an `[agrosynapse]` block into **Secrets**
   (see below). `requirements.txt` is read from the repo root automatically.
5. Pick a custom subdomain if you want one (for example `agrosynapse.streamlit.app`), then
   **Deploy**. The first build takes a few minutes while dependencies install.

Everything the dashboard needs (`africa_crop_yield.csv`, `clean_data.csv`, `features.csv`,
`predictions.csv`, `model_metrics.json`, `model_pipeline.joblib`) is committed, so the app
boots with a working model and data on first load. The model artefact is ~3 MB and loads in
well under a second.

### Demo mode (on by default)

Community Cloud serves every visitor from **one container sharing one filesystem**, so anything
written to disk is seen by everyone. The app therefore starts in **demo mode**:

- **Visitors can upload their own CSV and explore it.** The file is parsed into that visitor's
  browser session, never written to disk, and invisible to anyone else. The Dashboard and Data
  Explorer switch to it until the visitor picks *Back to demo data* or closes the tab. Uploads
  are capped at 25 MB (`.streamlit/config.toml`) and 250,000 rows.
- **Saving files, retraining the model and changing settings are disabled.** Retraining
  rewrites the model and processed data every visitor sees, so it cannot be per-visitor as
  built. The Model Performance page always describes the demo model, and says so while an
  upload is being explored.

`tests/test_uploads.py` checks these properties against the running app: an upload leaves
`data/` untouched, a second session cannot see it, and the shared-state buttons are disabled.

To allow writes:

```bash
# Local development - full access
AGROSYNAPSE_MODE=full streamlit run streamlit_app.py
```

```toml
# .streamlit/secrets.toml (gitignored) - hosted, with an operator unlock in the sidebar
[agrosynapse]
admin_password = "choose-something-long"
```

With `admin_password` set, the sidebar shows an **Operator access** box; entering the password
unlocks write actions for that browser session only. Setting `mode = "full"` in the same
secrets block disables demo mode entirely — do that only on a deployment nobody else can reach.

Note that Community Cloud's filesystem is **ephemeral**: anything written at runtime is lost
when the app sleeps or redeploys. Uploaded data and retrained models do not persist.

### FastAPI service

Community Cloud only serves the Streamlit app. To publish `app_api.py`, deploy it separately
(Render, Fly.io, Railway, or a container):

```bash
uvicorn app_api:app --host 0.0.0.0 --port ${PORT:-8000}
```

The service loads `data/processed/model_pipeline.joblib` relative to the repository root and
adds `scripts/` to `sys.path` so the pickled custom transformers (`ToNumericTransformer`,
`ToStringTransformer`) resolve. Ship `scripts/` alongside the model or the service will fail
to start. `/predict` requires every column the loaded pipeline was trained on — for the demo
model that is the 16 features listed in `model_metrics.json` under `features_used`.

---


## Data Sources

The demo dataset is assembled from two public, openly licensed sources:

| Source | What it provides | Licence |
|--------|------------------|---------|
| [FAOSTAT](https://www.fao.org/faostat/) - Production: Crops and Livestock Products (Africa bulk export) | Yield, area harvested and production per country / crop / year | CC BY 4.0 |
| [World Bank](https://data.worldbank.org/) - World Development Indicators | Country-year fertilizer use, land use, rural population, agricultural employment, GDP per capita, rural electrification, precipitation | CC BY 4.0 |

`scripts/build_dataset.py` downloads both, joins them on a normalised country name and year,
attaches previous-season history columns, and writes `data/raw/africa_crop_yield.csv`. Both
licences require attribution; the script records it in its module docstring, and the field-level
schema is documented in [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

---

## Documentation

| Document | Purpose |
|----------|---------|
| [PRD.md](docs/PRD.md) | Product vision and requirements |
| [UI_SPEC.md](docs/UI_SPEC.md) | Dashboard design spec |
| [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Field definitions for the demo panel and the survey template |

---

## Testing and Quality

```bash
# Install dev tooling
pip install -r requirements-dev.txt

# Run the test suite
pytest -q

# Lint and format
black scripts/ app_api.py
flake8 scripts/ app_api.py
```

The suite covers the parts that fail silently rather than loudly:

| File | What it protects |
|------|------------------|
| `tests/test_feature_config.py` | The allowlist fallback, and that identifier and target-derived columns never reach the model |
| `tests/test_build_dataset.py` | Country-name normalisation across FAOSTAT and World Bank spellings, and that the committed CSV still matches the builder's schema |
| `tests/test_train_model.py` | That the forward-looking split never sees future rows, and that the carry-forward benchmark is built only from training-period data |
| `tests/test_analysis.py` | That the quality checks catch real survey problems without flagging clean data, that outliers are judged within their crop, and that every sentence in the written summary matches the numbers |
| `tests/test_uploads.py` | That a visitor's upload stays in their session, never touches `data/`, and cannot be seen by another session; that retrain/save stay disabled in demo mode; that the analysis pages work on an uploaded survey; and that every page renders on the demo data |

The dashboard uses `st.navigation`, so each page has its own URL (`/crop-insights`, `/data-quality`, `/model-performance`, `/data-explorer`, `/uploads`, `/settings`, `/about`). To smoke-test that every page renders without raising:

```bash
pytest -q tests/test_uploads.py -k every_page
```

---


## Roadmap

- [x] FastAPI inference endpoint (`/health`, `/predict`)
- [x] pytest suite
- [x] Data quality report and Crop Insights pages
- [ ] Continuous integration (run the test suite on every push)
- [ ] Mixed-effects models for repeated measures (place as a random effect)
- [ ] Sub-national data for Nigeria (state-level yields) to go below the national aggregate
- [ ] Seasonal rainfall and temperature features in place of the long-run average
