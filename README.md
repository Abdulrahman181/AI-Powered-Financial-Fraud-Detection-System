#  AI-Powered Financial Fraud Detection System

An end-to-end machine learning system for detecting fraudulent financial transactions using advanced feature engineering, ensemble learning, and risk-based decision strategies.

---

##  Overview

Fraud detection is a critical problem in financial systems due to its **rare occurrence but high impact**.

This project builds a **production-ready fraud detection system** that:
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

###  Train/Test Split
- Time-based split (no random shuffling)
- 80% training / 20% testing

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

All models achieved:
- **ROC-AUC ≈ 0.95**

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
- Train: **0.967**
- Test: **0.951**
- Gap: **0.016 → Good generalization**

---

##  Interpretation

- High Recall → Detect more fraud cases  
- High Precision → Reduce false alarms  
- System supports **business-driven decisions**

---

##  Risk Engine

Transactions are classified into:

- 🔴 HIGH RISK (≥ 0.99)
- 🟠 MEDIUM RISK (≥ 0.95)
- 🟢 LOW RISK (< 0.95)
