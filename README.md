# 🔍 Hidden Bank Charge Detector

---

## 📌 Overview

An end-to-end machine learning system that analyzes bank transactions  
and identifies hidden or undisclosed bank charges — such as ATM fees, minimum balance 
penalties, forex markups, SMS alert fees, debit card annual charges, and more.

---

## 🏗️ Architecture

```
Transactions
        ↓
[Stage 1] Parser
     - CSV: auto-detect header, normalize columns
        ↓
[Stage 2] Feature Engineering (65+ features)
     - NLP: 17-category keyword matching, description patterns
     - Amount: debit flags, log-scale, rounding, small-amount signals
     - Temporal: hour, day, month, quarter-end, FY-end, RBI cycle awareness
     - Channel: UPI/NEFT/ATM/SYSTEM/ACH encoding
     - TxnID: pattern-based signals (SAL_, ATM_, T-prefix)
     - Statistical: user-level z-score, daily transaction count
        ↓
[Stage 3] SMOTE Oversampling (handles ~2.5% class imbalance)
        ↓
[Stage 4] Detection Ensemble (Soft Voting)
     - Calibrated GBM (weight=0.40) — 400 estimators, depth=5
     - Calibrated RF  (weight=0.35) — 300 estimators, balanced
     - Calibrated MLP (weight=0.25) — 256→128→64 ReLU
        ↓
[Stage 5] IsolationForest Anomaly Detector
     - Catches novel charges with no keyword signals
        ↓
[Combined Score] = 0.70 × detection_prob + 0.30 × anomaly_score
        ↓
[Stage 6] Multi-class Charge Type Classifier (GBM, 17 categories)
        ↓
[Stage 7] Explainability + RBI Regulation References
        ↓
[Stage 8] Dispute Letter Generator
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (first time only)
```bash
python train_model.py --data data/bank_transactions.csv
```

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
hidden_charge_detector/
├── app.py                        # Main Streamlit UI
├── train_model.py                # Training entry point
├── requirements.txt
├── README.md
├── data/
│   └── bank_transactions.csv     # Training dataset (put your data here)
├── models/                       # Saved model artifacts (auto-created)
│   ├── scaler.pkl
│   ├── ensemble.pkl
│   ├── iso_scaler.pkl
│   ├── iso_model.pkl
│   ├── type_clf.pkl
│   ├── feature_cols.pkl
│   ├── charge_types.pkl
│   └── metrics.pkl
└── src/
    ├── __init__.py
    ├── feature_engineering.py    # 65+ feature extractors
    ├── model.py                   # ML pipeline: train + predict
    ├── parser.py         # CSV/PDF parser
    └── explainability.py          # Explanations + dispute letter
```


---

## 🔍 Detected Charge Categories (17)

| Category | Description |
|----------|-------------|
| `atm_usage_charges` | ATM withdrawal/decline fees |
| `forex_markup_charges` | International transaction markup |
| `minimum_balance_penalty` | AQB non-maintenance penalty |
| `account_maintenance_charges` | Annual maintenance charges |
| `sms_alert_charges` | SMS transaction alert fees |
| `debit_card_annual_charges` | Annual/renewal debit card fee |
| `debit_card_replacement_charges` | Card reissuance fee |
| `cheque_book_charges` | Cheque book issuance fee |
| `cheque_bounce_charges` | Cheque/NACH return charges |
| `locker_charges` | Safe deposit locker rent |
| `fund_transfer_charges` | NEFT/RTGS/IMPS service fee |
| `cash_handling_charges` | Cash deposit/withdrawal fee |
| `dormant_account_charges` | Inoperative account penalty |
| `account_closure_charges` | Pre-closure fee |
| `overdraft_insufficient_balance` | OD penalty/interest |
| `statement_documentation_charges` | Physical statement fee |

---


