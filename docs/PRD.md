# Product Requirements Document (PRD)

## AgroSynapse

### Overview
AgroSynapse is a data-analysis tool for agricultural datasets. It checks a dataset's quality,
describes it (group comparisons, trends, peer benchmarks, a written summary) and, where the data
supports it, trains and evaluates a yield-forecasting model. It ships with a public national
crop-yield panel for Africa and accepts CSV uploads, including KoboToolbox survey exports.

### Problem Statement
Agricultural data, whether from field surveys or national statistics, arrives with blanks,
inconsistent labels, unit slips and duplicates. Analysts spend much of their time finding and
fixing these before any analysis can be trusted, and summaries are often written by hand. The
tool shortens that loop: it surfaces problems first, then produces comparisons and findings
that are reproducible and tied to the numbers.

### Objectives
- Report data-quality problems in any tabular CSV, ranked by severity, with examples
- Compare groups (crops, regions, varieties) with uncertainty shown, not just averages
- Show trends over time and benchmark one place against its peers
- Write findings in plain language, computed from the data rather than generated free-form
- Train a yield model with leakage-aware features and evaluate it against a naive forecast
- Serve predictions through an API

### Target Users
- Data analysts and research assistants working with crop, survey or trial data
- Agronomists and programme staff who need a quick, defensible read of a dataset
- Extension and monitoring teams reviewing survey exports before analysis

### Functional Requirements
- CSV upload with delimiter and encoding detection; uploads stay in the visitor's session
- Data-quality report: missing values, duplicates, repeated keys, impossible negatives,
  percentages over 100, implausible years, numbers stored as text, inconsistent labels,
  extra whitespace, and extreme values judged within groups
- Crop Insights: group means with 95% confidence intervals and a Kruskal-Wallis test, trend
  lines with a per-year change, peer benchmarking over a recent window, and a written summary
- Focus on IITA mandate crops (cassava, yam, maize, cowpea, soybean, banana and plantain)
- Cleaning pipeline with reproducible rules
- Machine-learning pipeline with a forward-looking evaluation and a carry-forward benchmark
- FastAPI inference endpoint

### Data Processing Rules
- Normalize column names (lowercase, underscores)
- Drop KoboToolbox metadata fields
- Keep identifiers, personal data and target-derived columns out of model features
- Handle missing data inside the model pipeline (median for numeric, most frequent for categorical)
- Encode categorical values (one-hot for modeling)

### Non-Functional Requirements
- Python 3.11, with pinned dependencies matching the committed model
- Fast inference (<1 second per prediction)
- Read-only by default when hosted; nothing a visitor uploads is written to disk
- Tested: analysis functions, feature policy, evaluation and upload isolation

### Success Metrics
- Quality checks catch planted errors in test data and do not flag clean data
- Every sentence in the written summary can be traced to a computed number
- The yield model beats the carry-forward benchmark on held-out years
- All dashboard pages render on both the demo data and an uploaded survey

### Known Limits
- The demo data is national (country x crop x year); it cannot answer farm-level questions
- Rows for the same place repeat across years, so intervals and tests are descriptive;
  repeated-measures models are future work
- Automatic column detection is a best guess; users can override every choice
- Extreme-value checks flag values for review; they cannot tell a real value from an error
