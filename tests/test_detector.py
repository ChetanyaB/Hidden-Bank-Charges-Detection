"""Test cases for the Hidden Bank Charge Detector (run: pytest -v)."""
import io
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.explainability import compute_summary_stats, explain_transaction, generate_dispute_letter  # noqa: E402
from src.model import load_artifacts, predict  # noqa: E402
from src.parser import generate_sample_csv, parse_csv, parse_passbook  # noqa: E402


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts(os.path.join(ROOT, "models"))


@pytest.fixture(scope="module")
def sample_result(artifacts):
    df = pd.read_csv(io.StringIO(generate_sample_csv()))
    return predict(df, artifacts, threshold=0.40)


# ---------- parsing ----------
def test_tc01_bank_style_columns_are_normalised():
    csv = "Txn Date,Narration,Ref No,Withdrawal,Deposit,Closing Balance\n" \
          "05-02-2025,SMS ALERT CHG,111,15,,9985\n06-02-2025,SALARY,222,,50000,59985\n"
    df = parse_passbook(io.BytesIO(csv.encode()), "statement.csv")
    assert {"description", "amount", "balance", "transaction_id", "timestamp"} <= set(df.columns)
    assert list(df["amount"]) == [-15.0, 50000.0]          # debit/credit merged into signed amount


def test_tc02_header_row_detected_after_bank_preamble():
    csv = "STATE BANK OF INDIA\nAccount: XXXX1234\n\nDate,Description,Amount,Balance\n" \
          "01-03-2025,ATM CHARGE,-23.6,1000\n"
    df = parse_csv(io.StringIO(csv))
    assert len(df) == 1 and df.loc[0, "description"] == "ATM CHARGE"


def test_tc03_indian_dates_parsed_day_first():
    df = parse_csv(io.StringIO("Date,Description,Amount\n13-02-2025,GST,-1.62\n05-02-2025,GST,-1\n"))
    assert df["timestamp"].dt.month.tolist() == [2, 2]


def test_tc04_unsupported_file_rejected():
    with pytest.raises(ValueError):
        parse_passbook(io.BytesIO(b"\x00\x01binary"), "statement.xlsx")


# ---------- detection ----------
def test_tc05_known_charges_are_flagged(sample_result):
    flagged = sample_result.loc[sample_result["predicted_hidden_charge"] == 1, "description"].tolist()
    for desc in ["ATM USAGE ADJ 3086704", "SMS ADJ 2331937", "MIN BAL CHRG JAN24", "NACH RTN ADJ 2432631"]:
        assert desc in flagged


def test_tc06_normal_spending_not_flagged(sample_result):
    normal = sample_result.set_index("description").loc[
        ["SALARY CREDIT JAN 2024", "UPI/9876543210/Swiggy", "POS/AMAZON INDIA"], "predicted_hidden_charge"]
    assert normal.sum() == 0


def test_tc07_charge_type_assigned(sample_result):
    types = dict(zip(sample_result["description"], sample_result["predicted_charge_type"]))
    assert types["SMS ADJ 2331937"] == "sms_alert_charges"
    assert types["MIN BAL CHRG JAN24"] == "minimum_balance_penalty"
    assert types["SALARY CREDIT JAN 2024"] == "none"


def test_tc08_numeric_transaction_ids_supported(artifacts):
    # crashed before the fix ("Can only use .str accessor with string values")
    df = pd.DataFrame({"timestamp": ["2025-01-19"], "description": ["SMS ALERT CHG"],
                       "transaction_id": [6789012345], "amount": [-15.0], "balance": [129312.0]})
    assert predict(df, artifacts)["predicted_hidden_charge"].iloc[0] == 1


def test_tc09_lower_threshold_flags_at_least_as_many(artifacts):
    df = pd.read_csv(io.StringIO(generate_sample_csv()))
    strict = predict(df, artifacts, 0.70)["predicted_hidden_charge"].sum()
    loose = predict(df, artifacts, 0.30)["predicted_hidden_charge"].sum()
    assert loose >= strict


def test_tc10_scores_and_risk_levels_valid(sample_result):
    assert sample_result["combined_score"].between(0, 1).all()
    assert set(sample_result["risk_level"]) <= {"🔴 Critical", "🟠 High", "🟡 Medium", "🟢 Low"}


# ---------- explainability & outputs ----------
def test_tc11_explanation_has_reasons(sample_result):
    row = sample_result[sample_result["description"] == "MIN BAL CHRG JAN24"].iloc[0]
    exp = explain_transaction(row)
    assert exp and isinstance(exp, dict)


def test_tc12_summary_totals_match(sample_result):
    s = compute_summary_stats(sample_result)
    flagged = sample_result[sample_result["predicted_hidden_charge"] == 1]
    assert s["flagged_count"] == len(flagged)
    assert s["hidden_charge_total"] == pytest.approx(flagged["amount"].abs().sum(), rel=1e-6)


def test_tc13_dispute_letter_lists_charges(sample_result):
    flagged = sample_result[sample_result["predicted_hidden_charge"] == 1]
    letter = generate_dispute_letter(flagged, account_holder="Test User",
                                     account_number="XXXXXXXX1234", bank_name="HDFC Bank")
    assert "HDFC Bank" in letter and "Test User" in letter and "600" in letter


def test_tc14_monthly_trend_populated(sample_result):
    trend = compute_summary_stats(sample_result)["monthly_trend"]
    assert not trend.empty and trend["count"].sum() == (sample_result["predicted_hidden_charge"] == 1).sum()
