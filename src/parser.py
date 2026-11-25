
import io
import re
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─── Column name normalization mappings ─────────────────────────────────────
COL_ALIASES = {
    "date": ["date", "date/time", "txn date", "transaction date", "value date",
             "posting date", "trans date", "timestamp"],
    "description": ["description", "narration", "particulars", "transaction details",
                    "details", "remarks", "txn desc", "transaction description",
                    "narrative", "desc"],
    "amount": ["amount", "txn amount", "transaction amount", "debit/credit",
               "dr/cr", "transaction amt"],
    "debit": ["debit", "dr", "withdrawal", "withdrawals", "debit amount"],
    "credit": ["credit", "cr", "deposit", "deposits", "credit amount"],
    "balance": ["balance", "closing balance", "avl balance", "available balance",
                "bal", "running balance", "ledger balance"],
    "transaction_id": ["transaction id", "txn id", "ref no", "reference number",
                       "chq no", "cheque no", "utr", "transaction no", "txn ref",
                       "ref number"],
    "transaction_channel": ["channel", "mode", "payment mode", "transaction mode",
                            "medium", "txn type"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names to standard names using alias lookup."""
    col_lower = {c: c.strip().lower() for c in df.columns}
    rename_map = {}
    for std_name, aliases in COL_ALIASES.items():
        for col, col_l in col_lower.items():
            if col_l in aliases and std_name not in rename_map.values():
                rename_map[col] = std_name
                break
    df = df.rename(columns=rename_map)
    return df


def merge_debit_credit(df: pd.DataFrame) -> pd.DataFrame:
    """If separate debit/credit columns exist, merge into a single signed 'amount'."""
    if "amount" not in df.columns:
        if "debit" in df.columns and "credit" in df.columns:
            df["debit"]  = pd.to_numeric(df["debit"].astype(str).str.replace(
                            r"[^\d.\-]", "", regex=True), errors="coerce").fillna(0)
            df["credit"] = pd.to_numeric(df["credit"].astype(str).str.replace(
                            r"[^\d.\-]", "", regex=True), errors="coerce").fillna(0)
            df["amount"] = df["credit"] - df["debit"]
        elif "debit" in df.columns:
            df["amount"] = -pd.to_numeric(df["debit"].astype(str).str.replace(
                            r"[^\d.\-]", "", regex=True), errors="coerce").fillna(0)
        elif "credit" in df.columns:
            df["amount"] = pd.to_numeric(df["credit"].astype(str).str.replace(
                            r"[^\d.\-]", "", regex=True), errors="coerce").fillna(0)
    return df


def clean_numeric(series: pd.Series) -> pd.Series:
    """Clean currency strings to float."""
    return pd.to_numeric(
        series.astype(str)
              .str.replace(r"[^\d.\-]", "", regex=True)
              .str.strip()
              .replace("", np.nan),
        errors="coerce"
    )


def parse_csv(file_obj) -> pd.DataFrame:
    """Parse a CSV file with auto-detection of header row."""
    try:
        # Try reading with header detection
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        lines = raw.splitlines()

        # Find the header row (first row with multiple comma-separated non-numeric fields)
        header_idx = 0
        for i, line in enumerate(lines[:20]):
            parts = line.split(",")
            non_num = sum(1 for p in parts if not re.match(r"^[\d\.\-\s]*$", p.strip()))
            if len(parts) >= 3 and non_num >= 2:
                header_idx = i
                break

        content = "\n".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(content), skip_blank_lines=True, dtype=str)
        df = df.dropna(how="all").reset_index(drop=True)
        df = normalize_columns(df)
        df = merge_debit_credit(df)

        # Clean numerics
        for col in ["amount", "balance"]:
            if col in df.columns:
                df[col] = clean_numeric(df[col])

        # Standardize timestamp
        if "date" in df.columns and "timestamp" not in df.columns:
            # Indian statements use DD-MM-YYYY, so parse day-first
            df["timestamp"] = pd.to_datetime(df["date"], dayfirst=True,
                                              errors="coerce")
            df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now())

        # Fill missing optional cols
        for col in ["transaction_id", "transaction_channel", "user_id"]:
            if col not in df.columns:
                df[col] = "UNKNOWN" if col != "user_id" else "USER001"

        # Drop rows where both amount and balance are NaN
        df = df.dropna(subset=["amount"], how="all").reset_index(drop=True)
        return df

    except Exception as e:
        raise ValueError(f"CSV parsing failed: {e}")


def parse_pdf_text(file_obj) -> pd.DataFrame:
    """
    Parse text-based PDF passbook using pdfplumber.
    Falls back to line-by-line regex parsing.
    """
    try:
        import pdfplumber
        rows = []
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        rows.extend(table)

        if rows:
            header = [str(h).strip() if h else f"col_{i}"
                      for i, h in enumerate(rows[0])]
            data   = rows[1:]
            df = pd.DataFrame(data, columns=header)
            df = df.dropna(how="all").reset_index(drop=True)
            df = normalize_columns(df)
            df = merge_debit_credit(df)
            for col in ["amount", "balance"]:
                if col in df.columns:
                    df[col] = clean_numeric(df[col])
            if "date" in df.columns and "timestamp" not in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"],
                                                  errors="coerce")
            for col in ["transaction_id", "transaction_channel", "user_id"]:
                if col not in df.columns:
                    df[col] = "UNKNOWN" if col != "user_id" else "USER001"
            df = df.dropna(subset=["amount"]).reset_index(drop=True)
            return df

    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: regex-based line parser
    return _parse_pdf_regex(file_obj)


def _parse_pdf_regex(file_obj) -> pd.DataFrame:
    """
    Regex-based fallback for PDF text extraction.
    Tries to find date + description + amount + balance patterns.
    """
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file_obj)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        text = ""

    # Pattern: date  description  amount  balance
    date_pat = r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{2}\s+\w{3}\s+\d{4})"
    amt_pat  = r"([\-]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"

    rows = []
    for line in text.splitlines():
        line = line.strip()
        m = re.search(date_pat, line)
        if not m:
            continue
        amounts = re.findall(amt_pat, line)
        amounts_clean = [float(a.replace(",", "")) for a in amounts if a]
        if len(amounts_clean) < 1:
            continue
        date_str = m.group(0)
        desc_start = m.end()
        desc_end   = line.rfind(amounts[-1]) if amounts else len(line)
        description = line[desc_start:desc_end].strip()

        amount  = amounts_clean[-2] if len(amounts_clean) >= 2 else amounts_clean[0]
        balance = amounts_clean[-1] if len(amounts_clean) >= 2 else 0.0

        rows.append({
            "timestamp":  date_str,
            "description": description,
            "amount":     amount,
            "balance":    balance,
            "transaction_id": f"PDF_{len(rows):05d}",
            "transaction_channel": "UNKNOWN",
            "user_id": "USER001",
        })

    if not rows:
        raise ValueError("Could not extract transactions from PDF. "
                         "Please use a CSV export from your bank.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"],
                                      errors="coerce")
    df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now())
    return df


def parse_passbook(file_obj, filename: str) -> pd.DataFrame:
    """
    Main entry: auto-detect format and parse.
    Returns a standardized DataFrame.
    """
    fname = filename.lower()
    if fname.endswith(".csv"):
        df = parse_csv(file_obj)
    elif fname.endswith(".pdf"):
        df = parse_pdf_text(file_obj)
    else:
        # Try CSV first
        try:
            df = parse_csv(file_obj)
        except Exception:
            raise ValueError(f"Unsupported file format: {filename}. "
                             "Please upload a CSV or PDF file.")

    # Validate required columns
    if "amount" not in df.columns:
        raise ValueError("Could not find 'Amount' column in passbook. "
                         "Please check your file format.")
    if "description" not in df.columns:
        df["description"] = "UNKNOWN"

    return df


def generate_sample_csv() -> str:
    """Generate a sample CSV for demo/testing."""
    rows = [
        ["2024-01-05 10:00:00", "SALARY CREDIT JAN 2024", "SAL_U001_202401", 55000.00, 60000.00, "NEFT", "success"],
        ["2024-01-08 14:22:00", "UPI/9876543210/Swiggy", "T100001", -340.00, 59660.00, "UPI", "success"],
        ["2024-01-10 09:00:00", "ATM USAGE ADJ 3086704", "T100002", -21.89, 59638.11, "SYSTEM", "success"],
        ["2024-01-12 11:30:00", "NEFT/HDFC000012345/Rent", "T100003", -15000.00, 44638.11, "NEFT", "success"],
        ["2024-01-15 08:00:00", "SMS ADJ 2331937", "T100004", -18.83, 44619.28, "SYSTEM", "success"],
        ["2024-01-18 16:00:00", "CARD ANN FEE 6553158", "T100005", -422.75, 44196.53, "SYSTEM", "success"],
        ["2024-01-20 12:00:00", "POS/AMAZON INDIA", "T100006", -2499.00, 41697.53, "POS", "success"],
        ["2024-01-22 09:00:00", "MIN BAL CHRG JAN24", "T100007", -600.00, 41097.53, "SYSTEM", "success"],
        ["2024-01-25 10:00:00", "INTL TXN FEE 9470650", "T100008", -187.02, 40910.51, "SYSTEM", "success"],
        ["2024-01-28 15:00:00", "NACH RTN ADJ 2432631", "T100009", -386.97, 40523.54, "SYSTEM", "success"],
    ]
    header = ["timestamp", "description", "transaction_id", "amount", "balance",
              "transaction_channel", "transaction_status"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(v) for v in r))
    return "\n".join(lines)
