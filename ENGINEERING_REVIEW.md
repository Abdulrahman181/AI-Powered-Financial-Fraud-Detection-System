# Engineering Review — AI-Powered Financial Fraud Detection System

**Repository:** `Abdulrahman181/AI-Powered-Financial-Fraud-Detection-System`
**Review scope:** all tracked Python/Notebook/README/dependency/config content at the review commit.
**Review date:** 2026-10-14
**Execution mode:** local sandbox; no GitHub push performed.

## Executive summary

The repository originally contained one large Colab-oriented notebook, a dataset URL, a short unpinned requirements file, and README claims tied to stored notebook outputs. The notebook included old execution outputs, Colab shell/upload cells, hard-coded local assumptions, and a Kaggle credential in a stored output. Its operating threshold was selected on the same chronological holdout used for reporting precision/recall, so those operating-point numbers were not an untouched final estimate. The notebook also changed `train_df`/`test_df` after `X_train`/`X_test` had already been materialized, making the intended feature removal ineffective, and it saved models without the feature state, column order, or selected threshold required for safe inference.

The implementation is now centered on `fraud_detection.py`. It validates real input data, fails clearly when data is absent, uses a 60/20/20 chronological train/validation/final-test split, fits feature statistics on training data only, selects the threshold on validation only, and stores feature metadata with locally generated artifacts. The notebook was rebuilt as a short, output-free walkthrough that delegates to this module. No PaySim data, trained models, or new performance result was fabricated or committed.

## Findings and remediation

### Critical / high priority

1. **Credential exposure in the original notebook output.** The prior notebook output contained a `kaggle.json` object with a Kaggle key. The current notebook output is empty and no credential is present in the working tree, but the old value remains in Git history because history rewriting and credential rotation are outside this local code change. **Action required:** revoke/rotate the exposed Kaggle key and, if policy requires, scrub historical Git objects before any public redistribution.
2. **Evaluation leakage through threshold selection.** The original code selected `best_threshold` from `X_test`/`y_test` and then reported precision/recall on that same holdout. The new workflow selects the F1 threshold on the validation period and evaluates once on a separate final test period. Average precision is reported alongside ROC-AUC because fraud is highly imbalanced.
3. **No data or artifacts were available for a full run.** Only `Data.txt` with the upstream Kaggle URL was present. The review did not download data, train the ensemble, or claim new model metrics. The CLI now stops with a clear `FileNotFoundError` and download instructions instead of inventing data.

### Medium priority

4. **Colab and path fragility.** The original notebook depended on `google.colab`, upload state, `!pip`, `!kaggle`, `!unzip`, and a CSV in the current Colab directory. These cells were removed. The path is now passed by `--data` or `PAYSIM_CSV`, with a documented repository-relative default.
5. **Feature-state and inference mismatch.** The original notebook did not persist quantiles, user/destination aggregates, feature order, threshold, or model version alongside its three model files. The new training routine writes model files plus `metadata.json` containing the feature order, training-fitted state, selected-on period, metrics, and seed. A production serving layer should still add artifact versioning, signatures, calibration, monitoring, and access control.
6. **Feature engineering order bug.** The old notebook dropped columns from `train_df`/`test_df` after `X_train`/`X_test` had been created, so the drop did not affect the matrices actually trained. The new module defines one `FEATURE_COLUMNS` contract and returns that exact order for every period.
7. **History aggregates.** Training user history is computed from prior rows only. Validation and test rows use frozen training-period aggregates rather than statistics from the full dataset. Negative downsampling is limited to training; validation and final test retain their natural distributions.
8. **API/error handling concerns.** The old Gradio function accepted loosely validated values and returned raw exception text in the response. That deployment cell has been removed from the canonical notebook; the shared module validates schema and numeric feature construction. Any future HTTP/UI adapter should validate finite/range-bounded inputs, avoid returning exception internals, authenticate requests, and load metadata together with models.

## Files changed

- `fraud_detection.py` — new reusable validation, split, feature engineering, optional deterministic ensemble training, threshold selection, evaluation, artifact metadata, and CLI.
- `Financial_Fraud_Detection_System.ipynb` — rebuilt with no outputs, credentials, shell magics, Colab upload calls, or absolute local paths.
- `README.md` — corrected methodology and reproducibility claims; documents the 60/20/20 split, validation-only threshold selection, data absence behavior, and commands.
- `requirements.txt` — bounded core runtime/training dependencies; removed unused UI-only dependencies from the canonical pipeline.
- `requirements-dev.txt` — pytest, Ruff, and nbformat tooling.
- `.gitignore` — excludes data, credentials, model artifacts, environments, caches, and notebook noise.
- `tests/test_pipeline.py` and `tests/__init__.py` — lightweight tests for missing data/schema, chronological boundaries, train-only feature state, fixed feature order, and threshold bounds.
- `ENGINEERING_REVIEW.md` — this report.

## Validation performed

- `python3 -m compileall -q fraud_detection.py tests` — passed.
- `ruff check Financial_Fraud_Detection_System.ipynb fraud_detection.py tests` — passed.
- `pytest -q` — **5 passed** after installing the declared scikit-learn dependency for the local test environment.
- Notebook JSON/AST validation — passed; 11 cells, zero outputs, zero execution counts, code parses successfully.
- Repository scan — no PaySim CSV, model artifact, Kaggle credential, Colab upload call, shell magic, `/content`, `/mnt`, or `share=True` remains in the canonical implementation/notebook. Documentation and `.gitignore` references to `kaggle.json` are intentional.
- Missing-data CLI check — exited non-zero with an actionable PaySim download/path message.

## Not run / limitations

- No full dataset was available, so the full 6M-row feature build, XGBoost/LightGBM/CatBoost training, artifact reload, calibration, latency, memory, and final test metrics were **not run**.
- No claim is made that the historical README AUC/precision/recall values are reproducible by this revised pipeline. They were removed as current performance claims.
- Dependency resolution was not performed for every heavy model package in this sandbox. The pinned ranges are declared, but a clean environment should install `requirements.txt` and run the full workflow with sufficient memory.
- The repository does not yet provide a production HTTP API. Before deployment, add a small authenticated adapter around the shared feature/artifact contract, request limits, structured safe errors, drift monitoring, and an independently held-out temporal test policy.
