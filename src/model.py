
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    VotingClassifier, IsolationForest
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

from src.feature_engineering import build_feature_matrix, get_feature_matrix, FEATURE_COLS


# ──────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────

def build_models():
    """
    Returns a dict of named sklearn pipelines.
    Each pipeline = StandardScaler → Classifier
    GBM captures non-linear interactions via boosting.
    RF provides low-variance bagging ensemble.
    MLP learns latent charge embeddings via deep features.
    """
    gbm = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )

    models = {
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", gbm),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", rf),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", mlp),
        ]),
    }
    return models


def build_stacking_ensemble(X_train, y_train):
    """
    Soft-voting ensemble of calibrated GBM + RF + MLP.
    Calibration ensures reliable probability estimates.
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.calibration import CalibratedClassifierCV

    gbm = GradientBoostingClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        min_samples_leaf=10, subsample=0.8, max_features="sqrt",
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu",
        solver="adam", learning_rate_init=0.001, max_iter=300,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=42,
    )

    # Calibrate each base learner
    cal_gbm = CalibratedClassifierCV(gbm, cv=3, method="isotonic")
    cal_rf  = CalibratedClassifierCV(rf,  cv=3, method="isotonic")
    cal_mlp = CalibratedClassifierCV(mlp, cv=3, method="sigmoid")

    ensemble = VotingClassifier(
        estimators=[("gbm", cal_gbm), ("rf", cal_rf), ("mlp", cal_mlp)],
        voting="soft",
        weights=[0.4, 0.35, 0.25],
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    ensemble.fit(X_scaled, y_train)
    return scaler, ensemble


def build_anomaly_detector(X_train):
    """
    IsolationForest for unsupervised anomaly scoring.
    Catches hidden charges with no keyword signals.
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.03,   # ~3% expected hidden charges
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    iso.fit(X_scaled)
    return scaler, iso


# ──────────────────────────────────────────────────────────
# Multi-class charge type classifier
# ──────────────────────────────────────────────────────────

CHARGE_TYPES = [
    "atm_usage_charges", "forex_markup_charges", "minimum_balance_penalty",
    "account_maintenance_charges", "sms_alert_charges", "debit_card_annual_charges",
    "cheque_book_charges", "cheque_bounce_charges", "locker_charges",
    "fund_transfer_charges", "cash_handling_charges", "dormant_account_charges",
    "account_closure_charges", "overdraft_insufficient_balance",
    "statement_documentation_charges", "debit_card_replacement_charges",
]

def build_type_classifier():
    """GBM multi-class classifier for charge type."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )),
    ])


def _smote_oversample(X: np.ndarray, y: np.ndarray,
                       k_neighbors: int = 5, random_state: int = 42) -> tuple:
    """
    Custom SMOTE implementation (no imblearn dependency).
    Generates synthetic minority-class samples via linear interpolation
    between a sample and one of its k nearest neighbours in feature space.
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(random_state)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y

    majority_cls = classes[np.argmax(counts)]
    minority_cls = classes[np.argmin(counts)]
    X_maj = X[y == majority_cls]
    X_min = X[y == minority_cls]

    n_synthetic = len(X_maj) - len(X_min)
    if n_synthetic <= 0:
        return X, y

    # Cap k to minority sample count
    k = min(k_neighbors, len(X_min) - 1)
    if k < 1:
        # Fall back to simple duplication with tiny noise
        repeats  = int(np.ceil(n_synthetic / len(X_min)))
        X_synth  = np.tile(X_min, (repeats + 1, 1))[:n_synthetic]
        X_synth += rng.normal(0, 1e-6, X_synth.shape)
    else:
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_min)
        nbrs = nn.kneighbors(X_min, return_distance=False)[:, 1:]  # exclude self

        synthetic = []
        for _ in range(n_synthetic):
            idx    = rng.randint(0, len(X_min))
            nn_idx = nbrs[idx, rng.randint(0, k)]
            alpha  = rng.random()
            synthetic.append(X_min[idx] + alpha * (X_min[nn_idx] - X_min[idx]))
        X_synth = np.array(synthetic)

    X_res = np.vstack([X, X_synth])
    y_res = np.concatenate([y, np.full(n_synthetic, minority_cls, dtype=y.dtype)])
    perm  = rng.permutation(len(X_res))
    return X_res[perm], y_res[perm]


# ──────────────────────────────────────────────────────────
# Training entry point
# ──────────────────────────────────────────────────────────

def train(df: pd.DataFrame, save_dir: str = "models") -> dict:
    """
    Full training pipeline:
    1. Feature engineering
    2. SMOTE oversampling for class imbalance
    3. Binary detection ensemble (GBM+RF+MLP soft vote)
    4. IsolationForest anomaly scorer
    5. Multi-class charge-type classifier
    6. Cross-validation metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    print("[1/6] Engineering features...")
    df_feat = build_feature_matrix(df, for_training=True)
    X = get_feature_matrix(df_feat).values
    y_binary = df["is_hidden_charge"].values.astype(int)

    # Multi-class labels
    y_type = df["hidden_charge_type"].fillna("none").values
    mask_hidden = y_binary == 1

    print(f"      Dataset: {len(X)} samples, {y_binary.sum()} hidden charges "
          f"({100*y_binary.mean():.1f}%)")

    # ── Binary detection ────────────────────────────────────
    print("[2/6] SMOTE oversampling...")
    X_res, y_res = _smote_oversample(X, y_binary, k_neighbors=5, random_state=42)
    print(f"      After SMOTE: {len(X_res)} samples")

    print("[3/6] Training detection ensemble (GBM+RF+MLP)...")
    scaler, ensemble = build_stacking_ensemble(X_res, y_res)

    # ── Anomaly detection ────────────────────────────────────
    print("[4/6] Training IsolationForest anomaly detector...")
    iso_scaler, iso_model = build_anomaly_detector(X)

    # ── Multi-class charge type classifier ───────────────────
    print("[5/6] Training charge-type multi-class classifier...")
    X_hidden = X[mask_hidden]
    y_hidden_type = y_type[mask_hidden]
    type_clf = build_type_classifier()
    type_scaler = StandardScaler()
    X_hidden_scaled = type_scaler.fit_transform(X_hidden)
    type_clf.named_steps["scaler"] = type_scaler
    type_clf.fit(X[mask_hidden], y_hidden_type)

    # ── Cross-validation ─────────────────────────────────────
    print("[6/6] Cross-validation (binary detection)...")
    from sklearn.pipeline import Pipeline as SKPipeline
    cv_pipe = SKPipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42,
        ))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_pipe, X_res, y_res, cv=cv,
                                scoring="roc_auc", n_jobs=-1)
    print(f"      5-Fold ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final train-set predictions for metrics
    X_scaled = scaler.transform(X)
    y_pred   = ensemble.predict(X_scaled)
    y_prob   = ensemble.predict_proba(X_scaled)[:, 1]

    # Anomaly scores (normalized 0-1, higher=more anomalous)
    X_iso   = iso_scaler.transform(X)
    iso_scores = -iso_model.score_samples(X_iso)   # Higher = more anomalous
    iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)

    # Combined score = 0.7 * detection_prob + 0.3 * anomaly_score
    combined_scores = 0.7 * y_prob + 0.3 * iso_scores

    metrics = {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std":  float(cv_scores.std()),
        "train_roc_auc":   float(roc_auc_score(y_binary, y_prob)),
        "train_f1":        float(f1_score(y_binary, y_pred)),
        "train_precision": float(precision_score(y_binary, y_pred)),
        "train_recall":    float(recall_score(y_binary, y_pred)),
        "n_samples":       int(len(X)),
        "n_hidden":        int(y_binary.sum()),
    }
    print(f"\n      Train ROC-AUC  : {metrics['train_roc_auc']:.4f}")
    print(f"      Train F1       : {metrics['train_f1']:.4f}")
    print(f"      Train Precision: {metrics['train_precision']:.4f}")
    print(f"      Train Recall   : {metrics['train_recall']:.4f}")

    # Save all artifacts
    artifacts = {
        "scaler": scaler,
        "ensemble": ensemble,
        "iso_scaler": iso_scaler,
        "iso_model": iso_model,
        "type_clf": type_clf,
        "feature_cols": FEATURE_COLS,
        "charge_types": CHARGE_TYPES,
        "metrics": metrics,
    }
    for name, obj in artifacts.items():
        with open(os.path.join(save_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)

    print(f"\n✅ All models saved to '{save_dir}/'")
    return metrics


def load_artifacts(save_dir: str = "models") -> dict:
    """Load all saved model artifacts."""
    artifacts = {}
    required = ["scaler", "ensemble", "iso_scaler", "iso_model",
                "type_clf", "feature_cols", "charge_types", "metrics"]
    for name in required:
        fpath = os.path.join(save_dir, f"{name}.pkl")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Model artifact not found: {fpath}. Train first.")
        with open(fpath, "rb") as f:
            artifacts[name] = pickle.load(f)
    return artifacts


def predict(df: pd.DataFrame, artifacts: dict,
            threshold: float = 0.40) -> pd.DataFrame:
    """
    Run full inference pipeline on new transactions.
    Returns original df with added columns:
      - hidden_charge_prob
      - anomaly_score
      - combined_score
      - predicted_hidden_charge  (0/1)
      - predicted_charge_type
      - risk_level  (Low/Medium/High/Critical)
    """
    df_feat = build_feature_matrix(df, for_training=False)
    X = get_feature_matrix(df_feat).values

    scaler   = artifacts["scaler"]
    ensemble = artifacts["ensemble"]
    iso_sc   = artifacts["iso_scaler"]
    iso_mod  = artifacts["iso_model"]
    type_clf = artifacts["type_clf"]

    X_scaled  = scaler.transform(X)
    y_prob    = ensemble.predict_proba(X_scaled)[:, 1]

    X_iso     = iso_sc.transform(X)
    iso_raw   = -iso_mod.score_samples(X_iso)
    iso_scores = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

    combined = 0.7 * y_prob + 0.3 * iso_scores
    predicted = (combined >= threshold).astype(int)

    # Charge type for flagged transactions
    charge_types = ["none"] * len(df)
    flagged_idx  = np.where(predicted == 1)[0]
    if len(flagged_idx) > 0:
        X_flagged = X[flagged_idx]
        type_preds = type_clf.predict(X_flagged)
        for i, idx in enumerate(flagged_idx):
            charge_types[idx] = type_preds[i]

    # Risk level
    def risk_level(score):
        if score >= 0.80:  return "🔴 Critical"
        if score >= 0.60:  return "🟠 High"
        if score >= 0.40:  return "🟡 Medium"
        return "🟢 Low"

    out = df.copy()
    out["hidden_charge_prob"]      = np.round(y_prob, 4)
    out["anomaly_score"]           = np.round(iso_scores, 4)
    out["combined_score"]          = np.round(combined, 4)
    out["predicted_hidden_charge"] = predicted
    out["predicted_charge_type"]   = charge_types
    out["risk_level"]              = [risk_level(s) for s in combined]

    return out
