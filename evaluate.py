"""Honest evaluation of the Hidden Charge Detector (run: python evaluate.py).

1. Held-out test: retrain the full pipeline on 80% of data/bank_transactions.csv
   (stratified) and score the untouched 20%. The metrics saved by train_model.py
   are measured on the training data, so they overstate real performance.
2. Real-statement check: run the shipped models (models/) on the 31 hand-labelled
   rows from a real-format bank passbook (same rows as bank_transaction_tests.py).

Outputs: docs/evaluation/metrics.json, docs/evaluation/*.png
"""
import io
import json
import os
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score, average_precision_score)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.model import load_artifacts, predict, train  # noqa: E402
from src.parser import parse_csv  # noqa: E402

OUT = os.path.join(ROOT, "docs", "evaluation")
THRESHOLD = 0.40

REAL_STATEMENT = """date,description,transaction_id,amount,balance,hidden_charge
13-02-2025,UPI/IMPSUPI,512345678,-18060,81814,0
27-02-2025,GST,78990343,-1.62,81812,1
24-01-2025,BY CASH FRM,604845678,4500,67890,0
30-01-2025,OSMS CHA,34567898765,-10,80967,1
11-07-2025,DR TRFR TO NEXT GEN,5678978,50000,190780,0
17-04-2025,CASH HAN,345678909,-100,87654,1
09-08-2025,GST 18% ON Charge,323000000000,-36,345876,1
20-11-1930,CONSOLIDATED CHARGES FOR A/C,76543456789,-200,765467,0
13-01-2025,ATM WITHDRAWAL,4567890123,-5000,77150,0
14-01-2025,NEFT SALARY CREDIT,1234567890,45000,122150,0
16-01-2025,GST,5678901234,-2.5,121827,1
17-01-2025,BY CASH DEP,3456789012,10000,131827,0
18-01-2025,IMPS TRANSFER,2345678901,-2500,129327,0
19-01-2025,SMS ALERT CHG,6789012345,-15,129312,1
20-01-2025,ATM WITHDRAWAL,7890123456,-2000,127312,0
21-01-2025,UPI/GROCERY,8901234567,-850,126462,0
22-01-2025,MIN BAL CHG,9012345678,-150,126312,1
23-01-2025,NEFT TRANSFER,1122334455,-10000,116312,0
24-01-2025,UPI/RECHARGE,2233445566,-199,116113,0
25-01-2025,CASH DEPOSIT,3344556677,5000,121113,0
26-01-2025,ATM CHARGE,4455667788,-23.6,121089,1
27-01-2025,UPI/OLA,5566778899,-240,120849,0
28-01-2025,RTGS TRANSFER,6677889900,-20000,100849,0
29-01-2025,INTEREST CREDIT,7788990011,120,100969,0
30-01-2025,DEBIT CARD FEE,8899001122,-250,100719,1
31-01-2025,UPI/NETFLIX,9900112233,-649,100070,0
"""


def binary_metrics(y, pred, score):
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, score), 4),
        "pr_auc": round(average_precision_score(y, score), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def holdout():
    df = pd.read_csv(os.path.join(ROOT, "data", "bank_transactions.csv"))
    tr, te = train_test_split(df, test_size=0.2, stratify=df["is_hidden_charge"], random_state=42)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        train(tr.reset_index(drop=True), save_dir=tmp)
        train_s = time.time() - t0
        arts = load_artifacts(tmp)
    t0 = time.time()
    res = predict(te.drop(columns=["is_hidden_charge", "hidden_charge_type"]).reset_index(drop=True), arts, THRESHOLD)
    ms_per_txn = (time.time() - t0) * 1000 / len(te)
    y = te["is_hidden_charge"].values
    m = binary_metrics(y, res["predicted_hidden_charge"].values, res["combined_score"].values)
    # Charge-type accuracy on correctly detected hidden charges
    hit = (y == 1) & (res["predicted_hidden_charge"].values == 1)
    true_types = te["hidden_charge_type"].values[hit]
    m["charge_type_accuracy"] = round(float((res["predicted_charge_type"].values[hit] == true_types).mean()), 4)
    m.update(n_train=len(tr), n_test=len(te), n_test_hidden=int(y.sum()),
             train_seconds=round(train_s, 1), inference_ms_per_txn=round(ms_per_txn, 3))
    return m, y, res


def real_statement():
    # Feed the rows through the same CSV parser the app uses for uploads
    labelled = pd.read_csv(io.StringIO(REAL_STATEMENT))
    y = labelled.pop("hidden_charge").values
    df = parse_csv(io.StringIO(labelled.to_csv(index=False)))
    res = predict(df, load_artifacts(os.path.join(ROOT, "models")), THRESHOLD)
    m = binary_metrics(y, res["predicted_hidden_charge"].values, res["combined_score"].values)
    m["n_rows"] = len(df)
    m["n_hidden"] = int(y.sum())
    res["true"] = y
    m["errors"] = res.loc[res["true"] != res["predicted_hidden_charge"],
                          ["description", "amount", "combined_score", "true", "predicted_hidden_charge"]] \
        .to_dict(orient="records")
    return m, y, res


def main():
    os.makedirs(OUT, exist_ok=True)
    print("== Held-out test (retraining on 80%, this takes a few minutes) ==")
    hm, hy, hres = holdout()
    print(json.dumps({k: v for k, v in hm.items()}, indent=1))
    print("\n== Real-statement sample (shipped models) ==")
    rm, ry, rres = real_statement()
    print(json.dumps(rm, indent=1, default=str))

    json.dump({"threshold": THRESHOLD, "held_out_test": hm, "real_statement_sample": rm},
              open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=str)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, y, res, title in [(ax[0], hy, hres, f"Held-out test ({len(hy):,} txns)"),
                             (ax[1], ry, rres, f"Real-statement sample ({len(ry)} rows)")]:
        ConfusionMatrixDisplay(confusion_matrix(y, res["predicted_hidden_charge"]),
                               display_labels=["Normal", "Hidden charge"]).plot(ax=a, colorbar=False, cmap="Blues")
        a.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "confusion_matrices.png"), dpi=110)

    from sklearn.metrics import precision_recall_curve
    fig, a = plt.subplots(figsize=(6, 4.5))
    for y, res, lab in [(hy, hres, "Held-out test"), (ry, rres, "Real-statement sample")]:
        p, r, _ = precision_recall_curve(y, res["combined_score"])
        a.plot(r, p, label=f"{lab} (AP={average_precision_score(y, res['combined_score']):.2f})")
    a.set_xlabel("Recall"); a.set_ylabel("Precision"); a.set_title("Precision–Recall")
    a.legend(); a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pr_curve.png"), dpi=110)
    print(f"\nSaved metrics and plots to {OUT}")


if __name__ == "__main__":
    main()
