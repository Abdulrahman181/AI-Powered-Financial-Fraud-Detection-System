#  AI-Powered Financial Fraud Detection System

An end-to-end machine learning system for detecting fraudulent financial transactions using advanced feature engineering, ensemble learning, and risk-based decision strategies.

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

The ensemble achieved **0.951918 ROC-AUC** on the chronological holdout. This is reported as a holdout result, not as an untouched external test benchmark.

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
- System supports **business-driven decisions**

---

##  Risk Engine

The current prototype classifies transactions into:

- 🔴 HIGH RISK (≥ 0.99)
- 🟠 MEDIUM RISK (≥ 0.95)
- 🟢 LOW RISK (< 0.95)
