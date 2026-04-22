
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─── Indian-bank charge keyword dictionaries ────────────────────────────────
HIDDEN_CHARGE_KEYWORDS = {
    "atm_usage":        ["atm chrg", "atm fee", "atm usage", "atm chg", "atm decline", "atm fail",
                         "cash withdrawal fee", "atm adj", "atm transaction fee"],
    "forex_markup":     ["intl txn", "forex", "international", "foreign currency", "fx markup",
                         "intl fee", "cross border", "intl transaction fee"],
    "minimum_balance":  ["min bal", "minimum balance", "min balance", "non-maintenance", "aqb",
                         "avg quarterly", "qab penalty", "average balance"],
    "maintenance":      ["amc", "amc chg", "account maintenance", "acct maint", "maintenance charge",
                         "account fees", "annual maintenance"],
    "sms_alert":        ["sms", "sms alert", "sms adj", "message alert", "sms chrg",
                         "alert charges", "sms charges"],
    "debit_card":       ["card ann", "dc renewal", "card reissue", "debit card fee", "card fee",
                         "card maint", "card annual", "card issuance"],
    "cheque":           ["chqbk", "cheque book", "chq book", "chequebook", "nach rtn", "chq rtn",
                         "cheque bounce", "chq ret", "nach return", "ecs return"],
    "locker":           ["locker", "safe deposit", "locker fee", "safe custody"],
    "fund_transfer":    ["txn svc fee", "neft chg", "rtgs chg", "imps fee", "transfer fee",
                         "svc fee", "service charge", "neft fee"],
    "cash_handling":    ["cash dep fee", "cash handling", "cash deposit charge", "cash deposit fee",
                         "cash teller fee", "currency handling"],
    "dormant":          ["dormant", "drmnt", "inoperative", "dormancy", "inactive account"],
    "closure":          ["acct closure", "account closure", "closure charge", "exit fee",
                         "pre-closure"],
    "overdraft":        ["od chg", "od penalty", "overdraft", "od adj", "od fee",
                         "insufficient balance", "od interest"],
    "statement":        ["statement", "passbook", "doc charges", "statement charge",
                         "documentation fee", "physical statement"],
}

KEYWORD_TO_LABEL = {kw: category for category, kwlist in HIDDEN_CHARGE_KEYWORDS.items()
                    for kw in kwlist}

BANK_KEYWORDS = {
    "SBI":   ["sbi", "state bank", "yono"],
    "HDFC":  ["hdfc", "hdfc bank"],
    "ICICI": ["icici", "icici bank"],
    "AXIS":  ["axis", "axis bank"],
    "KOTAK": ["kotak", "kotak bank", "kotak mahindra"],
    "PNB":   ["pnb", "punjab national"],
    "BOB":   ["bob", "bank of baroda"],
    "CANARA":["canara", "canara bank"],
    "UNION": ["union bank", "uboi"],
    "IDFC":  ["idfc", "idfc first"],
}

SYSTEM_CHANNELS = {"SYSTEM", "ACH"}


def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract NLP-based features from description text."""
    df = df.copy()
    desc = df["description"].fillna("").str.lower()

    # Keyword match flags (one-hot for each charge category)
    for category, keywords in HIDDEN_CHARGE_KEYWORDS.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        df[f"kw_{category}"] = desc.str.contains(pattern, regex=True).astype(int)

    # Total keyword category hits
    kw_cols = [c for c in df.columns if c.startswith("kw_")]
    df["kw_total_hits"] = df[kw_cols].sum(axis=1)

    # Specific charge-like patterns in description (regex)
    df["has_adj_suffix"]     = desc.str.contains(r"\badj\b").astype(int)
    df["has_chg_suffix"]     = desc.str.contains(r"\bch[rg]+\b").astype(int)
    df["has_fee_suffix"]     = desc.str.contains(r"\bfee\b").astype(int)
    df["has_penalty_word"]   = desc.str.contains(r"\bpenalt\w*\b").astype(int)
    df["has_return_word"]    = desc.str.contains(r"\b(rtn|return|ret)\b").astype(int)
    df["has_number_in_desc"] = desc.str.contains(r"\d{5,}").astype(int)  # Long ID numbers in desc
    df["desc_len"]           = desc.str.len()
    df["desc_word_count"]    = desc.str.split().str.len().fillna(0)
    df["desc_upper_ratio"]   = (df["description"].fillna("").apply(
                                    lambda x: sum(1 for c in x if c.isupper())) /
                                (df["desc_len"] + 1))

    # System/bank-generated description indicators
    df["is_system_desc"] = desc.str.contains(
        r"\b(sys|system|auto|automatic|scheduled|standing|si |ecs|nach|neft adj|rtgs adj)\b",
        regex=True
    ).astype(int)

    return df


def extract_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Statistical and domain amount features."""
    df = df.copy()
    df["is_debit"]         = (df["amount"] < 0).astype(int)
    df["abs_amount"]       = df["amount"].abs()
    df["log_abs_amount"]   = np.log1p(df["abs_amount"])
    df["is_round_amount"]  = (df["abs_amount"] % 100 == 0).astype(int)
    df["is_very_small"]    = (df["abs_amount"] < 100).astype(int)       # Tiny fees
    df["is_micro"]         = (df["abs_amount"] < 50).astype(int)        # SMS / alert fees
    df["is_medium"]        = ((df["abs_amount"] >= 100) &
                               (df["abs_amount"] < 2000)).astype(int)
    df["amount_ends_99"]   = (df["abs_amount"] % 100 == 99).astype(int)

    # Balance-related features
    df["balance_after_txn"]   = df["balance"]
    df["low_balance_flag"]    = (df["balance"] < 5000).astype(int)
    df["very_low_balance"]    = (df["balance"] < 1000).astype(int)
    df["balance_went_neg"]    = (df["balance"] < 0).astype(int)

    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based features."""
    df = df.copy()
    try:
        dt = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception:
        dt = pd.to_datetime(df["timestamp"], errors="coerce")

    df["hour"]           = dt.dt.hour
    df["day_of_week"]    = dt.dt.dayofweek          # 0=Mon
    df["day_of_month"]   = dt.dt.day
    df["month"]          = dt.dt.month
    df["is_weekend"]     = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = (dt.dt.day <= 5).astype(int)
    df["is_month_end"]   = (dt.dt.day >= 25).astype(int)
    df["is_quarter_end"] = (dt.dt.month.isin([3, 6, 9, 12]) &
                            (dt.dt.day >= 25)).astype(int)
    df["is_year_end"]    = ((dt.dt.month == 3) & (dt.dt.day >= 28)).astype(int)  # Indian FY end
    df["is_off_hours"]   = ((df["hour"] < 9) | (df["hour"] >= 18)).astype(int)

    # Quarter (Indian FY: Apr–Mar)
    df["fy_quarter"] = ((dt.dt.month - 4) % 12 // 3 + 1).fillna(0).astype(int)

    return df


def extract_channel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction channel features."""
    df = df.copy()
    channel = df.get("transaction_channel", pd.Series(["UNKNOWN"] * len(df)))
    channel = channel.fillna("UNKNOWN").str.upper()

    df["channel_is_system"]      = channel.isin(SYSTEM_CHANNELS).astype(int)
    df["channel_is_atm"]         = (channel == "ATM").astype(int)
    df["channel_is_pos"]         = (channel == "POS").astype(int)
    df["channel_is_neft"]        = (channel == "NEFT").astype(int)
    df["channel_is_upi"]         = (channel == "UPI").astype(int)
    df["channel_is_debit_card"]  = (channel == "DEBIT_CARD").astype(int)
    df["channel_is_imps"]        = (channel == "IMPS").astype(int)
    df["channel_is_bbps"]        = (channel == "BBPS").astype(int)

    # Encode channel ordinally
    channel_order = {"UPI": 0, "NEFT": 1, "IMPS": 2, "POS": 3,
                     "ATM": 4, "DEBIT_CARD": 5, "BBPS": 6, "ACH": 7,
                     "SYSTEM": 8, "UNKNOWN": 9}
    df["channel_encoded"] = channel.map(channel_order).fillna(9).astype(int)

    return df


def extract_transaction_id_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract signals from transaction ID patterns."""
    df = df.copy()
    tid = df.get("transaction_id", pd.Series([""] * len(df))).fillna("").str.upper()

    df["txn_id_is_sal"]   = tid.str.startswith("SAL_").astype(int)   # Salary
    df["txn_id_is_atm"]   = tid.str.contains("ATM").astype(int)
    df["txn_id_is_t_prefix"] = tid.str.match(r"^T\d+").astype(int)
    df["txn_id_len"]      = tid.str.len()
    df["txn_id_has_date"] = tid.str.contains(r"\d{8}").astype(int)

    return df


def extract_statistical_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user rolling statistics and deviation features."""
    df = df.copy()

    if "user_id" not in df.columns:
        df["amount_vs_user_mean"] = 0.0
        df["amount_z_score"]      = 0.0
        df["txn_count_per_day"]   = 1
        return df

    try:
        dt = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception:
        dt = pd.to_datetime(df["timestamp"], errors="coerce")

    df["_date"] = dt.dt.date

    # User-level amount stats
    user_stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).reset_index()
    user_stats.columns = ["user_id", "user_mean_amt", "user_std_amt"]
    df = df.merge(user_stats, on="user_id", how="left")
    df["user_std_amt"] = df["user_std_amt"].fillna(1)
    df["amount_vs_user_mean"] = df["amount"] - df["user_mean_amt"]
    df["amount_z_score"]      = df["amount_vs_user_mean"] / (df["user_std_amt"] + 1e-6)

    # Transactions per day per user
    daily_cnt = df.groupby(["user_id", "_date"]).size().reset_index(name="txn_count_per_day")
    df = df.merge(daily_cnt, on=["user_id", "_date"], how="left")
    df["txn_count_per_day"] = df["txn_count_per_day"].fillna(1)

    df.drop(columns=["_date", "user_mean_amt", "user_std_amt"], inplace=True, errors="ignore")

    return df


def build_feature_matrix(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Master pipeline: run all feature extractors and return final feature matrix.
    """
    df = df.copy()

    # Rename columns to standard names if needed (handle CSV uploads )
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower().replace(" ", "_").replace("/", "_")
        col_map[c] = cl
    df.rename(columns=col_map, inplace=True)

    # Ensure required columns
    for req in ["description", "amount", "balance"]:
        if req not in df.columns:
            df[req] = 0 if req in ("amount", "balance") else ""

    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = df["date"]
    if "timestamp" not in df.columns and "date/time" in df.columns:
        df["timestamp"] = df["date/time"]
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now()

    # Run all feature extractors
    df = extract_text_features(df)
    df = extract_amount_features(df)
    df = extract_temporal_features(df)
    df = extract_channel_features(df)
    df = extract_transaction_id_features(df)
    df = extract_statistical_context_features(df)

    return df


FEATURE_COLS = [
    # Text keyword flags
    "kw_atm_usage", "kw_forex_markup", "kw_minimum_balance", "kw_maintenance",
    "kw_sms_alert", "kw_debit_card", "kw_cheque", "kw_locker",
    "kw_fund_transfer", "kw_cash_handling", "kw_dormant", "kw_closure",
    "kw_overdraft", "kw_statement", "kw_total_hits",
    # Text pattern flags
    "has_adj_suffix", "has_chg_suffix", "has_fee_suffix", "has_penalty_word",
    "has_return_word", "has_number_in_desc", "desc_len", "desc_word_count",
    "desc_upper_ratio", "is_system_desc",
    # Amount features
    "is_debit", "abs_amount", "log_abs_amount", "is_round_amount",
    "is_very_small", "is_micro", "is_medium", "amount_ends_99",
    "balance_after_txn", "low_balance_flag", "very_low_balance", "balance_went_neg",
    # Temporal features
    "hour", "day_of_week", "day_of_month", "month", "is_weekend",
    "is_month_start", "is_month_end", "is_quarter_end", "is_year_end",
    "is_off_hours", "fy_quarter",
    # Channel features
    "channel_is_system", "channel_is_atm", "channel_is_pos",
    "channel_is_neft", "channel_is_upi", "channel_is_debit_card",
    "channel_is_imps", "channel_is_bbps", "channel_encoded",
    # Transaction ID features
    "txn_id_is_sal", "txn_id_is_atm", "txn_id_is_t_prefix",
    "txn_id_len", "txn_id_has_date",
    # Statistical context
    "amount_vs_user_mean", "amount_z_score", "txn_count_per_day",
]


def get_feature_matrix(df_engineered: pd.DataFrame) -> pd.DataFrame:
    """Select and fill feature columns from engineered dataframe."""
    available = [c for c in FEATURE_COLS if c in df_engineered.columns]
    missing   = [c for c in FEATURE_COLS if c not in df_engineered.columns]
    if missing:
        for c in missing:
            df_engineered[c] = 0
    return df_engineered[FEATURE_COLS].fillna(0)
