#  AI-Powered Financial Fraud Detection System

An educational machine-learning prototype for exploring fraud detection on the synthetic PaySim dataset. It is not a production system or a tool for real financial decisions.

---

##  Overview

Fraud detection is a critical problem in financial systems due to its **rare occurrence but high impact**.

This project builds an **end-to-end fraud-detection prototype** that:
- Handles **extreme class imbalance (~0.13% fraud)**
- Learns **user behavior patterns**
- Uses **ensemble learning for robust predictions**
- Provides **risk-based decision outputs**

---

##  Dataset

  - Source: PaySim dataset
  - Size: 6.3 million transactions
  - Features: 11 original features
  - Data Quality:
    - No missing values
    - No duplicates

---

##  Key Challenges

- Extreme class imbalance
- Fraud is very rare (~0.13%)
- Need to balance:
  - High fraud detection (Recall)
  - Low false alarms (Precision)

---

##  Key Insights

- Fraud occurs mainly in:
  - `TRANSFER`
  - `CASH_OUT`
- High transaction amounts → higher fraud risk
- Fraud activity increases during night hours
- Each fraud transaction is typically unique per user

---

##  Data Processing

###  Cleaning
- Removed unnecessary balance-related features
- Log transformation applied to `amount`

###  Train/Holdout Split
- Chronological, step-ordered split with no random shuffling
- 80% training / 20% chronological holdout
- The holdout contains **1,272,524 transactions**

---

##  Feature Engineering

Engineered features focusing on **behavior + risk patterns**:

###  Time Features
- Hour of transaction
- Night indicator

###  Amount Features
- Log-transformed amount
- High / Very high transaction flags

###  User Behavior
- Transaction count
- Average transaction amount
- Last transaction amount

###  Relative Features
- Amount vs user average
- Ratio vs last transaction

###  Destination Features
- Merchant detection
- Destination popularity
- New destination flag

###  Risk Features
- High-risk transfer indicator

---

##  Handling Imbalance

- Downsampling of non-fraud cases
- `scale_pos_weight` for XGBoost / LightGBM
- `auto_class_weights` for CatBoost

---

##  Models Used

- XGBoost
- LightGBM
- CatBoost

The notebook records an ensemble ROC-AUC of **0.951918** on its chronological holdout. This figure was not independently reproduced in this repository update, and it is not an untouched external benchmark or evidence of performance on real transactions.

---

##  Ensemble System

A weighted ensemble combining all models:

- XGBoost → 30%
- LightGBM → 30%
- CatBoost → 40%

###  Why Ensemble?
- Improves stability
- Reduces variance
- Combines strengths of different models

---

##  Performance

###  ROC-AUC
- Training sample: **0.966423**
- Chronological holdout: **0.951918**
- Gap: **0.014505 AUC**, equivalent to **1.45 percentage points**

### Threshold Analysis

The notebook evaluates several operating points using precision-recall analysis. At the F1-selected threshold of **0.981419**, the holdout output reports **60.8% precision** and **36.4% recall**. Because that threshold was selected on the same holdout used for reporting, these precision/recall values are exploratory operating-point results—not an untouched final estimate. A production evaluation should select the threshold on a separate validation period and report once on a final chronological test period.

---

##  Interpretation

- High Recall → Detect more fraud cases  
- High Precision → Reduce false alarms  
- Outputs are for **educational review only**; they do not support or authorize real financial decisions

---

##  Risk Engine

The notebook uses the following exploratory score bands (the model scores are not calibrated probabilities):

- 🔴 HIGH RISK (≥ 0.99)
- 🟠 MEDIUM RISK (≥ 0.95)
- 🟢 LOW RISK (< 0.95)


## Safe setup, scope, and security

- The notebook is designed for Google Colab and the synthetic PaySim dataset. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab Secrets before running the data-download cells; do not paste credentials into a notebook cell, upload a credential file interactively, or commit credentials. The notebook writes a temporary `kaggle.json` in the runtime and does not display its contents.
- Install the packages in `requirements.txt` in a compatible runtime. Dependency versions are not pinned, so this repository does not guarantee bit-for-bit reproducibility. The dataset itself is not included; obtain it from the [PaySim Kaggle dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) and the original notebook output identified `CC-BY-SA-4.0`, but verify the current dataset license and attribution requirements directly on Kaggle before reuse or redistribution.
- PaySim is synthetic data. The recorded holdout metrics have not been re-run as part of this update, and do not establish performance on real transactions. Model outputs are **uncalibrated ensemble scores**, not probabilities or confidence values; the displayed risk bands are illustrative.
- **Credential incident:** a prior public notebook revision contained a Kaggle credential in saved output. This PR removes saved notebook outputs, but a normal PR cannot erase credentials from already-published Git history or other clones. Treat the credential as compromised: revoke it and create a replacement in Kaggle now. The repository owner should separately coordinate Git-history cleanup across affected refs and verify the public repository/cache state; do not add the replacement credential to Git.
- The demo launches with `share=False`. Do not expose it publicly or use real transaction/customer information.
