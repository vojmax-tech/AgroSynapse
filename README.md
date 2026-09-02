# AgroSynapse

> **Agricultural Survey Intelligence Platform** - Transforming agricultural survey data into predictive insights for smallholder farmers and extension services.

---

## Overview

**AgroSynapse** is a venture-stage data platform that combines mobile-first survey collection (KoboToolbox), ML-powered predictive analytics, and interactive dashboards to unlock productivity gains for farming communities. We transform any agricultural survey data into actionable insights for farm-level decisions.

### Why AgroSynapse?
- **350M+ smallholder farmers** lack access to data-driven recommendations
- **Mobile-first collection** reaches farmers where they are
- **ML models** trained on soil, weather, and farming practice data
- **Actionable dashboards** for extension agents and cooperatives

---

## Technology Stack

```text
KoboToolbox (Mobile Collection [Desktop web for training])
  -> Python ETL (Pandas / Scikit-learn)
  -> ML Models (RandomForest, Regression)
  -> Streamlit Dashboard (Analytics UI)
  -> FastAPI (Inference API)
```

---

## Key Features

- **Mobile-first surveys**: KoboToolbox integration and offline support
- **Automated ETL**: Metadata filtering, imputation, and one-hot encoding
- **Predictive models**: Machine learning models for any agricultural outcome prediction
- **Interactive dashboard**: Streamlit UI for extension agents and cooperatives
- **Fast inference**: Sub-second predictions on new survey submissions
- **Data privacy**: On-premise deployment with no cloud lock-in

---

## Project Structure

```text
agrosynapse/
|-- data/
|   |-- raw/                        # Kobo exports (input)
|   `-- processed/                  # clean_data.csv, features.csv,
|                                   # model_pipeline.joblib, model_metrics.json
|-- scripts/
|   |-- ingest_kobo.py
|   |-- clean_data.py
|   |-- feature_config.py
|   |-- transform_utils.py
|   |-- transform_features.py
|   |-- train_model.py
|   |-- predict.py
|   `-- run_pipeline.py
|-- streamlit_app.py                # Analytics dashboard
|-- app_api.py                      # FastAPI inference service
|-- assets/                         # Logo and dashboard CSS
|-- .streamlit/config.toml          # Dashboard theme
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

## Data Processing Rules

- Normalize column names (lowercase with underscores)
- Drop Kobo metadata fields
- Model pipeline handles imputation and encoding during training/inference

---

## ETL Usage

```bash
# 1) Ingest Kobo export
python scripts/ingest_kobo.py --input data/raw/kobo_export.csv
# Default output is data/raw/<input_stem>_raw.csv

# Optional: append timestamp for versioned ingest snapshots
python scripts/ingest_kobo.py --input data/raw/kobo_export.csv --timestamp-output

# 2) Clean data
python scripts/clean_data.py --input data/raw/kobo_export_raw.csv --output data/processed/clean_data.csv
# (or point --input to the timestamped file generated in step 1)

# 3) Train model
python scripts/train_model.py --input data/processed/clean_data.csv --target yield_tons --output data/processed/model_pipeline.joblib

# 3.5) Generate engineered features for explorer/model diagnostics
python scripts/transform_features.py --input data/processed/clean_data.csv --target yield_tons --output-dir data/processed

# 4) Predict
python scripts/predict.py --model data/processed/model_pipeline.joblib --input data/processed/clean_data.csv --output data/processed/predictions.csv

# Or run full pipeline
python scripts/run_pipeline.py

# Run full pipeline on a custom raw file
python scripts/run_pipeline.py --input data/raw/AgroSynapse_Survey_.csv
```

---

## Quick Start

### Prerequisites
- Python 3.11+ (the committed `model_pipeline.joblib` is built with scikit-learn 1.8)
- pip

### Installation

```bash
# Clone and setup
git clone https://github.com/victoryoma1/agrosynapse.git
cd agrosynapse

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

### Run the Full Pipeline

```bash
# 1. Export your Kobo survey
# File -> Download -> CSV, save as data/raw/kobo_export.csv

# 2. Execute end-to-end pipeline
python scripts/run_pipeline.py
# Or specify your export path directly
# python scripts/run_pipeline.py --input data/raw/AgroSynapse_Survey_.csv

# 3. Launch dashboard
streamlit run streamlit_app.py

# 4) Start inference API
uvicorn app_api:app --reload --host 0.0.0.0 --port 8000
```

Dashboard opens at: `http://localhost:8501`
API opens at: `http://localhost:8000`

Use the API from a mobile or web client by POSTing survey records to `http://localhost:8000/predict`.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [PRD.md](docs/PRD.md) | Product vision and requirements |
| [UI_SPEC.md](docs/UI_SPEC.md) | Dashboard design spec |
| [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Survey field definitions |

---

## Testing and Quality

```bash
# Install dev tooling
pip install -r requirements-dev.txt

# Lint and format
black scripts/ app_api.py
flake8 scripts/ app_api.py
```

> There is no automated test suite yet -- see the roadmap.

---

## Roadmap

- [ ] **v0.2**: FastAPI inference endpoint and mobile app integration (scaffolded)
- [ ] **v0.3**: Multi-language support (Hausa, Swahili, French)
- [ ] **v0.4**: Advanced analytics (time-series forecasting, anomaly detection)
- [ ] **v1.0**: Production-grade deployment and optional Postgres sync

---

## For Investors and Partners

- **Market**: 45M+ farms in Sub-Saharan Africa alone
- **Revenue models**: B2B2C licensing, per-prediction API calls, premium analytics
- **MVP status**: Functional predictive modeling platform with reproducible metrics in `data/processed/model_metrics.json`
- **Next milestone**: Deploy pilot with 10K+ farmers in 2 countries

For investor deck or partnership inquiries: investors@agrosynapse.io

---

## License

MIT License - see [LICENSE](LICENSE) for details.
