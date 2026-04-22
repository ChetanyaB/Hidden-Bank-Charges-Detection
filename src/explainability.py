
import pandas as pd
import numpy as np
from datetime import datetime
from src.feature_engineering import HIDDEN_CHARGE_KEYWORDS


# ─── Rule-based explanations ─────────────────────────────────────────────────

CHARGE_EXPLANATIONS = {
    "atm_usage_charges": (
        "ATM Usage Fee",
        "Banks like SBI, HDFC, ICICI charge ₹20–₹21 per ATM transaction after "
        "the free limit (typically 5 per month at own ATMs, 3 at other ATMs). "
        "RBI mandates this to be disclosed in MITC.",
        "Check your ATM usage against your bank's free transaction limit. "
        "Consider switching to a zero-fee savings account."
    ),
    "forex_markup_charges": (
        "Foreign Currency Markup Fee",
        "International transactions attract a forex markup of 1.5%–3.5% over the "
        "interbank rate. This is often not shown at point of sale. SEBI/RBI require "
        "disclosure but many customers miss it.",
        "Use zero-forex credit cards (e.g., Niyo Global, IndusInd) for international "
        "payments. File a complaint if not disclosed."
    ),
    "minimum_balance_penalty": (
        "Minimum Balance Non-Maintenance Charge",
        "Banks penalize ₹100–₹750/month if your Average Quarterly Balance (AQB) "
        "falls below the required threshold (₹5,000–₹25,000 for regular savings accounts). "
        "RBI mandates proportional penalty, not flat fee.",
        "Maintain AQB or switch to a zero-balance BSBD (Basic Savings Bank Deposit) account. "
        "File grievance if penalty exceeds 10% of shortfall."
    ),
    "account_maintenance_charges": (
        "Annual Account Maintenance Charge",
        "Charged for account upkeep, usually ₹150–₹750/year. Should be disclosed in account T&C. "
        "Some banks waive this for salary accounts or above a certain balance.",
        "Check if your account qualifies for waiver. Request waiver in writing if balance "
        "threshold maintained throughout the year."
    ),
    "sms_alert_charges": (
        "SMS Alert Fee",
        "Banks charge ₹15–₹25/quarter for SMS transaction alerts. RBI mandates that "
        "banks must offer free email alerts. SMS alerts should be opt-out.",
        "Opt for email alerts (free under RBI directive). Contact your branch to disable "
        "paid SMS service."
    ),
    "debit_card_annual_charges": (
        "Debit Card Annual/Renewal Fee",
        "Annual debit card fee ranges from ₹100–₹750 depending on card variant (Classic/Platinum/Business). "
        "Should be disclosed at issuance.",
        "Downgrade to a free variant or switch to a bank offering lifetime-free debit cards. "
        "Check if fee was auto-charged without reminder."
    ),
    "cheque_book_charges": (
        "Cheque Book Issuance Fee",
        "First cheque book (25 leaves) is usually free. Additional books charged ₹30–₹75. "
        "Some banks charge for each cheque leaf beyond the free limit.",
        "Go paperless. Use NEFT/IMPS/UPI for transfers instead of cheques."
    ),
    "cheque_bounce_charges": (
        "Cheque Return / NACH Return Charge",
        "Penalty for returned cheques (outward/inward) ranges ₹200–₹700 per instance. "
        "NACH/ECS return attracts ₹300–₹500. Lender may also charge.",
        "Ensure sufficient balance before ECS/NACH debit dates. Set up auto-top-up "
        "from linked accounts."
    ),
    "locker_charges": (
        "Safe Deposit Locker Fee",
        "Annual locker rent ranges ₹1,000–₹10,000+ depending on size and branch. "
        "RBI capped locker charges and mandated a new locker agreement since Jan 2022.",
        "Verify locker charges match your signed agreement. RBI allows you to surrender "
        "locker and claim proportional refund."
    ),
    "fund_transfer_charges": (
        "Fund Transfer Service Fee",
        "NEFT/RTGS charges were abolished by RBI for online transactions (Nov 2019). "
        "Branch-initiated NEFT still allowed to charge. IMPS can be charged up to ₹25.",
        "Always use internet/mobile banking for NEFT/RTGS (free). Avoid branch-initiated "
        "transfers."
    ),
    "cash_handling_charges": (
        "Cash Deposit / Handling Fee",
        "Free cash deposits limited to 3–5 per month for savings accounts. Beyond free "
        "limit, charges of ₹2–₹5 per ₹1,000 or flat ₹150 apply. Non-home branch deposits "
        "attract additional fees.",
        "Use home branch for deposits. Prefer digital transfers. SBI charges ₹2/₹1000 "
        "beyond ₹10,000 per day."
    ),
    "dormant_account_charges": (
        "Dormant Account Reactivation Charge",
        "Accounts inactive for 24 months are marked dormant. Banks charge ₹100–₹500 "
        "for reactivation. RBI circular states dormant accounts should NOT attract charges "
        "if the customer initiates reactivation by visiting branch.",
        "File a complaint with RBI BankingOmbudsman if dormant reactivation fee was "
        "charged — this may violate RBI guidelines."
    ),
    "account_closure_charges": (
        "Account Closure / Pre-closure Fee",
        "Closure within 14 days: free. Closure between 14 days–12 months: ₹300–₹500. "
        "After 12 months: free at most banks. Should be disclosed in account opening form.",
        "Wait 12 months before closing account to avoid fees. Get written confirmation "
        "of zero balance before closure."
    ),
    "overdraft_insufficient_balance": (
        "Overdraft / Insufficient Balance Penalty",
        "When ECS/NACH debits or charges push account below zero, overdraft interest "
        "(18%–30% p.a.) and penalty (₹200–₹500 per instance) is charged. "
        "Banks must inform customer before using overdraft facility.",
        "Set up SMS/email alerts for low balance. Maintain buffer above all auto-debit amounts."
    ),
    "statement_documentation_charges": (
        "Statement / Documentation Fee",
        "Physical statements charged ₹50–₹200. Digital statements are free under RBI directive. "
        "Duplicate statements charged ₹50–₹150 per statement.",
        "Always download statements from net banking (free). Opt for e-statement delivery."
    ),
    "debit_card_replacement_charges": (
        "Debit Card Replacement Fee",
        "Card reissuance (damaged, stolen, lost) costs ₹150–₹350. "
        "First replacement at some banks may be free.",
        "Report loss/theft immediately (reduces liability). Check if your account offers "
        "free first replacement."
    ),
}


def explain_transaction(row: pd.Series) -> dict:
    """Generate a structured explanation for a flagged transaction."""
    charge_type = row.get("predicted_charge_type", "none")
    prob        = row.get("combined_score", 0.0)
    amount      = row.get("amount", 0.0)
    description = row.get("description", "")

    if charge_type in CHARGE_EXPLANATIONS:
        title, why, action = CHARGE_EXPLANATIONS[charge_type]
    else:
        title  = "Unclassified Bank Charge"
        why    = ("This debit does not match a regular purchase pattern and shows "
                  "characteristics of a bank-imposed fee.")
        action = ("Review your bank's schedule of charges (SOC) document. Contact "
                  "customer care for clarification.")

    # Rule-based evidence bullets
    evidence = []
    desc_lower = description.lower()

    for category, keywords in HIDDEN_CHARGE_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                evidence.append(f"Description contains charge keyword: '{kw}'")
                break

    if abs(amount) < 100:
        evidence.append(f"Small debit amount ₹{abs(amount):.2f} — typical of service fees")
    elif abs(amount) < 1000:
        evidence.append(f"Moderate charge of ₹{abs(amount):.2f} — common fee range")

    if prob > 0.8:
        evidence.append(f"High model confidence: {prob:.0%}")
    elif prob > 0.6:
        evidence.append(f"Moderate-high model confidence: {prob:.0%}")

    channel = row.get("transaction_channel", "")
    if str(channel).upper() in ("SYSTEM", "ACH"):
        evidence.append("Transaction initiated by SYSTEM/ACH — bank-automated charge")

    if not evidence:
        evidence.append("Statistical anomaly vs your transaction history")

    return {
        "title": title,
        "amount": abs(amount),
        "why": why,
        "action": action,
        "evidence": evidence,
        "rbi_reference": _get_rbi_reference(charge_type),
    }


def _get_rbi_reference(charge_type: str) -> str:
    refs = {
        "atm_usage_charges":     "RBI/2021-22/125 — ATM Interchange Fee Structure",
        "forex_markup_charges":  "RBI Circular DPSS.CO.OD No.1023/02.01.001/2019-20",
        "minimum_balance_penalty": "RBI/2012-13/350 — Levy of Penal Charges on Non-maintenance of Min. Balance",
        "sms_alert_charges":     "RBI/2018-19/129 — Mandate for free email transaction alerts",
        "fund_transfer_charges": "RBI/2019-20/81 — NEFT/RTGS charges waived for online banking",
        "dormant_account_charges": "RBI Master Circular on Customer Service (DBOD No. Leg. BC.59/09.07.005/2012-13)",
        "locker_charges":        "RBI Circular DOR.LEG.REC.No.30/09.08.024/2021-22 — Safe Deposit Locker revision",
        "account_closure_charges": "RBI Guidelines on Account Closure, DBOD.No.Leg.BC.37/09.07.005/2003-04",
    }
    return refs.get(charge_type, "RBI Master Circular on Customer Service")


def generate_dispute_letter(flagged_rows: pd.DataFrame,
                             account_holder: str = "Account Holder",
                             account_number: str = "XXXXXXXXXXXX",
                             bank_name: str = "Your Bank") -> str:
    """Generate a formal dispute letter for submission to the bank."""
    today = datetime.now().strftime("%d %B %Y")
    total = flagged_rows["amount"].abs().sum()
    n_charges = len(flagged_rows)

    txn_lines = []
    for _, row in flagged_rows.iterrows():
        date_str = str(row.get("timestamp", "N/A"))[:10]
        desc     = str(row.get("description", "N/A"))
        amt      = abs(float(row.get("amount", 0)))
        ctype    = str(row.get("predicted_charge_type", "N/A")).replace("_", " ").title()
        txn_lines.append(f"  • {date_str} | {desc} | ₹{amt:,.2f} | {ctype}")

    txn_block = "\n".join(txn_lines)

    letter = f"""
Date: {today}

The Branch Manager
{bank_name}
[Branch Address]

Subject: Formal Dispute of Undisclosed / Unauthorized Bank Charges — Account No. {account_number}

Dear Sir/Madam,

I, {account_holder}, holder of Savings Account No. {account_number}, wish to formally dispute
the following {n_charges} charge(s) totaling ₹{total:,.2f} debited from my account, which I 
believe were either undisclosed, unauthorized, or in violation of RBI guidelines:

{txn_block}

I request the bank to:
1. Provide a detailed written explanation and Schedule of Charges reference for each debit above.
2. Reverse all charges that were not explicitly consented to or adequately disclosed at account opening.
3. Confirm in writing within 15 business days (as per RBI directive on complaint resolution).

If this matter is not resolved within 30 days, I reserve my right to:
• File a complaint with the Banking Ombudsman under RBI Integrated Ombudsman Scheme 2021.
• Escalate to the Consumer Disputes Redressal Commission under Consumer Protection Act 2019.

I request an acknowledgement of this letter and a complaint reference number.

Yours faithfully,

{account_holder}
Account No.: {account_number}
Date: {today}
Email: [Your Email]
Mobile: [Your Mobile]

---
Note: This letter was prepared with the assistance of the Hidden Charge Detection System.
All flagged transactions should be independently verified before filing.
"""
    return letter.strip()


def compute_summary_stats(df_result: pd.DataFrame) -> dict:
    """Compute analytics summary for the dashboard."""
    total_txns    = len(df_result)
    flagged       = df_result[df_result["predicted_hidden_charge"] == 1]
    n_flagged     = len(flagged)
    total_amount  = df_result["amount"].abs().sum()
    hidden_amount = flagged["amount"].abs().sum()

    charge_type_counts = (
        flagged["predicted_charge_type"]
        .value_counts()
        .to_dict()
    )

    monthly_hidden = pd.DataFrame()
    try:
        df_result["_month"] = pd.to_datetime(
            df_result["timestamp"], errors="coerce"
        ).dt.to_period("M").astype(str)
        monthly_hidden = (
            flagged.groupby("_month")["amount"]
            .agg(count="count", total=lambda x: x.abs().sum())
            .reset_index()
            .rename(columns={"_month": "month"})
        )
    except Exception:
        pass

    return {
        "total_transactions":  total_txns,
        "flagged_count":       n_flagged,
        "flagged_pct":         round(100 * n_flagged / max(total_txns, 1), 2),
        "total_debits":        round(total_amount, 2),
        "hidden_charge_total": round(hidden_amount, 2),
        "hidden_pct_of_debits": round(100 * hidden_amount / max(total_amount, 1), 2),
        "charge_type_breakdown": charge_type_counts,
        "monthly_trend":       monthly_hidden,
        "avg_charge":          round(flagged["amount"].abs().mean() if n_flagged else 0, 2),
        "max_charge":          round(flagged["amount"].abs().max() if n_flagged else 0, 2),
    }
