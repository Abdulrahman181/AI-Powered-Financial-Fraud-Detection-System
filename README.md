# AI-Powered Financial Fraud Detection System

Leakage-aware PaySim fraud-detection reference pipeline. The repository contains a reproducible Python workflow and a cleaned notebook; it does **not** contain the PaySim data or trained model artifacts.

## Data and responsible use

- **Dataset:** [PaySim1 on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) (the repository stores only the source URL in `Data.txt`).
- Download the CSV under the dataset terms and place it at `data/PS_20174392719_1491204439457_log.csv`, or pass another path with `--data` / `PAYSIM_CSV`.
- Do not commit `kaggle.json`, downloaded CSVs, model files, or real transaction data. The pipeline never downloads data and never creates a fallback dataset.
- This is a research prototype, not a production authorization or customer-decisioning system. Thresholds, calibration, drift, privacy, access control, and human review require a separate risk/compliance process.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m fraud_detection --data data/PS_20174392719_1491204439457_log.csv
```

The default command validates the file and prints its row count only. A missing file or invalid schema stops with an actionable error; no synthetic data is substituted.

To train locally (this may require substantial memory/time for the full PaySim file):

```bash
python -m fraud_detection --data data/PS_20174392719_1491204439457_log.csv \
  --train --output-dir artifacts
```

Training writes `artifacts/*.joblib` and `artifacts/metadata.json`, which are ignored by Git. The final test metrics are printed only after threshold selection on a separate chronological validation period.

To reload a complete bundle and score new, schema-valid rows without dropping the persisted feature state or threshold:

```python
from fraud_detection import predict_with_artifacts

predictions = predict_with_artifacts(frame, "artifacts")
# columns: probability, isFraud
```

`load_artifacts` checks the artifact version, exact feature order, persisted feature state, threshold, and all three model files before returning them. This is a local library contract, not an authenticated HTTP API.

## Methodology

1. Validate the required PaySim schema and non-empty, two-class labels.
2. Sort by `step` and split into approximately **60% chronological train / 20% validation / 20% final test** using whole `step` groups. No random split is used, and rows from one time step cannot cross a boundary.
3. Fit amount quantiles, user aggregates, and destination popularity on training data only. Training user history uses prior rows only; validation/test use frozen training-period aggregates.
4. Downsample negatives only in the training period. Validation and final test retain their natural class distribution.
5. Train seeded XGBoost, LightGBM, and CatBoost models with single-threaded inference settings for more stable reruns, then combine them with weights 0.3/0.3/0.4.
6. Select the operating threshold on validation and report it once on the untouched final test. Report ROC-AUC, average precision, precision, and recall; accuracy is intentionally not used as the primary fraud metric.

The notebook is an executable walkthrough of these functions. It contains no stored outputs, credentials, Colab upload calls, shell magics, or local absolute paths. It stops at data loading if the CSV is absent.

## Validation and development checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m compileall -q fraud_detection.py tests
pytest -q
```

The tests cover missing-data/schema and numeric-domain failures, whole-step chronological boundaries, train-only feature state and label alignment, fixed feature ordering, metric and sampling contracts, artifact metadata compatibility, and validation-only threshold selection. They do not claim a full model run without the real dataset and model dependencies.

## Repository layout

- `fraud_detection.py` — validation, leakage-aware feature engineering, optional training/evaluation CLI.
- `Financial_Fraud_Detection_System.ipynb` — cleaned, output-free walkthrough.
- `tests/` — lightweight tests that do not download or invent a training dataset for results.
- `Data.txt` — upstream dataset URL only.
- `ENGINEERING_REVIEW.md` — audit findings, changes, validation evidence, and limitations.
