import io, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

RAW_CSV = """\
date,description,transaction_id,amount,balance,hidden_charge
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

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(io.StringIO(RAW_CSV))
df["date"]    = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df["amount"]  = pd.to_numeric(df["amount"],  errors="coerce").fillna(0)
df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0)
y  = df["hidden_charge"].astype(int).values
df = df.drop(columns=["hidden_charge"])

# ── features (numeric only) ───────────────────────────────────────────────────
df["abs_amount"]   = df["amount"].abs()
df["is_debit"]     = (df["amount"] < 0).astype(int)
df["small_amount"] = (df["abs_amount"] < 100).astype(int)
df["month"]        = df["date"].dt.month.fillna(0).astype(int)
df["day"]          = df["date"].dt.day.fillna(0).astype(int)
df["log_balance"]  = np.log1p(df["balance"].abs())
df["year_bad"]     = ((df["date"].dt.year < 2020) | (df["date"].dt.year > 2026)).astype(int)

FEAT_COLS = ["abs_amount","is_debit","small_amount","month","day","log_balance","year_bad"]
X = df[FEAT_COLS].values.astype(float)

# ── models ────────────────────────────────────────────────────────────────────
models = {
    "RandomForest"     : RandomForestClassifier(n_estimators=100, max_depth=3,
                                                class_weight="balanced", random_state=42),
    "GradientBoosting" : GradientBoostingClassifier(n_estimators=50, max_depth=2,
                                                     learning_rate=0.1, random_state=42),
    "LogisticRegression": Pipeline([("sc", StandardScaler()),
                                    ("lr", LogisticRegression(C=0.1,
                                     class_weight="balanced", max_iter=500))]),
}

loo  = LeaveOneOut()
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

loo_res, skf_res = {}, {}
for name, clf in models.items():
    for res, cv in [(loo_res, loo), (skf_res, skf5)]:
        yp  = cross_val_predict(clf, X, y, cv=cv)
        res[name] = {
            "y_pred"   : yp,
            "accuracy" : accuracy_score(y, yp),
            "f1"       : f1_score(y, yp, zero_division=0),
            "precision": precision_score(y, yp, zero_division=0),
            "recall"   : recall_score(y, yp, zero_division=0),
        }

best = max(loo_res, key=lambda k: loo_res[k]["accuracy"])

# ── feature importance ────────────────────────────────────────────────────────
rf_full = RandomForestClassifier(n_estimators=200, max_depth=4,
                                  class_weight="balanced", random_state=42)
rf_full.fit(X, y)
feat_imp = pd.Series(rf_full.feature_importances_, index=FEAT_COLS).sort_values()

# ── console output ────────────────────────────────────────────────────────────
print("\n  LOO-CV Results")
print(f"  {'Model':<22} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
print(f"  {'-'*50}")
for name, r in loo_res.items():
    print(f"  {name:<22} {r['accuracy']:>6.3f} {r['f1']:>6.3f} "
          f"{r['precision']:>6.3f} {r['recall']:>6.3f}")

print(f"\n  Classification Report — {best} (LOO-CV):")
print(classification_report(y, loo_res[best]["y_pred"],
                             target_names=["Normal","Hidden_Charge"], zero_division=0))

print("  Missed rows:")
for i, (desc, yt, yp) in enumerate(zip(df["description"], y, loo_res[best]["y_pred"])):
    if yt != yp:
        print(f"    Row {i:>2}: {desc}  (true={yt}, pred={yp})")

# ── plots ─────────────────────────────────────────────────────────────────────
os.makedirs("/mnt/user-data/outputs", exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Hidden Charge Detection — LOO-CV Evaluation", fontsize=14, fontweight="bold")

model_names = list(loo_res.keys())
x = np.arange(len(model_names)); w = 0.35

# (A) Accuracy — LOO vs 5-Fold
ax = axes[0, 0]
loo_accs = [loo_res[m]["accuracy"] for m in model_names]
skf_accs = [skf_res[m]["accuracy"] for m in model_names]
b1 = ax.bar(x - w/2, loo_accs, width=w, label="LOO-CV",    color="#5C6BC0", edgecolor="white")
b2 = ax.bar(x + w/2, skf_accs, width=w, label="5-Fold CV", color="#26A69A", edgecolor="white")
ax.axhline(0.80, color="red",  linestyle="--", linewidth=1.2, label="80%")
ax.axhline(0.90, color="navy", linestyle="--", linewidth=1.2, label="90%")
ax.set_xticks(x)
ax.set_xticklabels([m.replace("Regression","Reg.") for m in model_names], fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy by CV Strategy", fontweight="bold")
ax.legend(fontsize=8)
for b, v in zip(list(b1)+list(b2), loo_accs+skf_accs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

# (B) F1 — LOO vs 5-Fold
ax = axes[0, 1]
loo_f1s = [loo_res[m]["f1"] for m in model_names]
skf_f1s = [skf_res[m]["f1"] for m in model_names]
b1 = ax.bar(x - w/2, loo_f1s, width=w, label="LOO-CV",    color="#EF5350", edgecolor="white")
b2 = ax.bar(x + w/2, skf_f1s, width=w, label="5-Fold CV", color="#FF9800", edgecolor="white")
ax.axhline(0.80, color="red",  linestyle="--", linewidth=1.2)
ax.axhline(0.90, color="navy", linestyle="--", linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels([m.replace("Regression","Reg.") for m in model_names], fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("F1"); ax.set_title("F1 Score by CV Strategy", fontweight="bold")
ax.legend(fontsize=8)
for b, v in zip(list(b1)+list(b2), loo_f1s+skf_f1s):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

# (C) Feature importance
ax = axes[0, 2]
feat_imp.plot(kind="barh", ax=ax, color="#7E57C2", edgecolor="white")
ax.set_title("Feature Importance (RF)", fontweight="bold")
ax.set_xlabel("Importance")

# (D-F) Confusion matrices per model
for col, name in enumerate(model_names):
    ax = axes[1, col]
    cm = confusion_matrix(y, loo_res[name]["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Normal","HC"]).plot(ax=ax, colorbar=False, cmap="Blues")
    r  = loo_res[name]
    ax.set_title(f"{name}\nAcc={r['accuracy']:.3f}  F1={r['f1']:.3f}  "
                 f"Prec={r['precision']:.3f}  Rec={r['recall']:.3f}",
                 fontweight="bold", fontsize=9)

plt.tight_layout()
out = "/mnt/user-data/outputs/hidden_charge_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")

# ── per-row heatmap ───────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 7))
best_preds = loo_res[best]["y_pred"]
desc_labels = [f"{i}: {d[:25]}" for i, d in enumerate(df["description"])]
colors = ["#43A047" if yt == yp else "#EF5350"
          for yt, yp in zip(y, best_preds)]
bars = ax2.barh(desc_labels, [1]*len(y), color=colors, edgecolor="white", height=0.7)
for i, (yt, yp) in enumerate(zip(y, best_preds)):
    label = f"True={yt}  Pred={yp}" + ("  ✗ MISS" if yt != yp else "")
    ax2.text(0.5, i, label, ha="center", va="center",
             fontsize=8.5, color="white", fontweight="bold")
ax2.set_xlim(0, 1)
ax2.set_xlabel("")
ax2.tick_params(axis="x", bottom=False, labelbottom=False)
ax2.tick_params(axis="y", labelsize=8.5)
ax2.set_title(f"Per-row Predictions — {best} (LOO-CV)\n"
              f"Green = correct   Red = missed", fontweight="bold", fontsize=11)
plt.tight_layout()
out2 = "data/hidden_charge_perrow.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"\n  Charts saved.")