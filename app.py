
import os
import sys
import io
import pickle
import warnings
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.parser import parse_passbook, generate_sample_csv
from src.model import predict, load_artifacts, train
from src.explainability import explain_transaction, generate_dispute_letter, compute_summary_stats
from src.feature_engineering import build_feature_matrix, get_feature_matrix

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hidden Charge Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
    color: white;
}
.main-header h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
.main-header p  { font-size: 1rem; opacity: 0.85; margin-top: 0.4rem; }

.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #0f3460;
    margin-bottom: 0.8rem;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #0f3460; }
.metric-label { font-size: 0.82rem; color: #666; margin-top: 0.2rem; }

.alert-critical { border-left: 4px solid #e74c3c; background: #fdf2f0; padding: 0.8rem; border-radius: 6px; margin: 0.4rem 0; }
.alert-high     { border-left: 4px solid #e67e22; background: #fef9f0; padding: 0.8rem; border-radius: 6px; margin: 0.4rem 0; }
.alert-medium   { border-left: 4px solid #f1c40f; background: #fefdf0; padding: 0.8rem; border-radius: 6px; margin: 0.4rem 0; }
.alert-low      { border-left: 4px solid #27ae60; background: #f0fdf4; padding: 0.8rem; border-radius: 6px; margin: 0.4rem 0; }

.evidence-pill {
    display: inline-block;
    background: #e8f4fd;
    color: #1a5276;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px;
}
.pipeline-step {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.9rem;
    border-left: 3px solid #0f3460;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
MODEL_DIR   = "models"
DATA_PATH   = "data/bank_transactions.csv"
MODELS_EXIST = all(
    os.path.exists(os.path.join(MODEL_DIR, f"{n}.pkl"))
    for n in ["scaler", "ensemble", "iso_scaler", "iso_model", "type_clf", "feature_cols"]
)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Hidden Charge Detector")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Home & Upload",
        "📊 Analysis Dashboard",
        "📄 Dispute Letter Generator",
    ])

    st.divider()
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.20, max_value=0.80,
        value=0.40, step=0.05,
        help="Lower = more sensitive (more detections, may include false positives)"
    )

    if MODELS_EXIST:
        st.success("✅ Models loaded")
    else:
        st.warning("⚠️ Models not trained yet\nGo to Model Training tab")


# ─── Session state ───────────────────────────────────────────────────────────
if "results"    not in st.session_state: st.session_state.results    = None
if "raw_df"     not in st.session_state: st.session_state.raw_df     = None
if "artifacts"  not in st.session_state: st.session_state.artifacts  = None
if "summary"    not in st.session_state: st.session_state.summary    = None


def load_models_cached():
    if st.session_state.artifacts is None and MODELS_EXIST:
        try:
            st.session_state.artifacts = load_artifacts(MODEL_DIR)
        except Exception as e:
            st.error(f"Model loading failed: {e}")
    return st.session_state.artifacts


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: Home & Upload
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Upload":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Hidden Bank Charge Detector</h1>
        <p>AI-powered analysis to identify undisclosed fees & charges</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">17+</div>
            <div class="metric-label">Charge Categories Detected</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3-Stage</div>
            <div class="metric-label">ML Pipeline (GBM + RF + MLP + IsolationForest)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">RBI</div>
            <div class="metric-label">Regulation-Aware Explanations</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📤 Upload CSV")
    st.info("Supported formats: **CSV** \n\n")

    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload",
            type=["csv"],
            help="Export your bank statement as CSV from net banking",
        )

    with col_demo:
        st.markdown("**Or try with sample data:**")
        if st.button("📋 Load Sample Data", use_container_width=True):
            sample_csv = generate_sample_csv()
            df_sample  = pd.read_csv(io.StringIO(sample_csv))
            st.session_state.raw_df = df_sample
            st.success("Sample data loaded!")

    if uploaded is not None:
        with st.spinner("Parsing..."):
            try:
                df_parsed = parse_passbook(uploaded, uploaded.name)
                st.session_state.raw_df = df_parsed
                st.success(f"✅ Parsed {len(df_parsed):,} transactions from '{uploaded.name}'")
            except Exception as e:
                st.error(f"Parsing failed: {e}")
                st.info("💡 Tip: Export your statement as CSV from net banking for best results.")

    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.markdown("#### Preview (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Total: {len(df):,} transactions | "
                   f"Columns: {', '.join(df.columns.tolist())}")

        if MODELS_EXIST:
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                artifacts = load_models_cached()
                with st.spinner("Running ML pipeline..."):
                    progress = st.progress(0, "Feature engineering...")
                    result_df = predict(df, artifacts, threshold=threshold)
                    progress.progress(70, "Computing summary...")
                    summary   = compute_summary_stats(result_df)
                    progress.progress(100, "Done!")
                    st.session_state.results = result_df
                    st.session_state.summary = summary
                st.success(f"✅ Analysis complete! Found **{summary['flagged_count']}** "
                           f"potential hidden charges (₹{summary['hidden_charge_total']:,.2f})")
                st.info("👈 Navigate to **Analysis Dashboard** to see results.")
        else:
            st.warning("⚠️ Models not yet trained. Go to **Model Training** tab first.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: Analysis Dashboard
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Analysis Dashboard":
    st.markdown("## 📊 Analysis Dashboard")

    if st.session_state.results is None:
        st.info("No analysis results yet. Upload a CSV and run analysis first.")
        st.stop()

    result_df = st.session_state.results
    summary   = st.session_state.summary

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{summary['total_transactions']:,}")
    c2.metric("Hidden Charges Found", f"{summary['flagged_count']}",
              delta=f"{summary['flagged_pct']}% of total", delta_color="inverse")
    c3.metric("Total Hidden Amount", f"₹{summary['hidden_charge_total']:,.2f}",
              delta="vs total debits", delta_color="inverse")
    c4.metric("Avg Charge", f"₹{summary['avg_charge']:,.2f}")
    c5.metric("Largest Charge", f"₹{summary['max_charge']:,.2f}")

    st.divider()

    # Charts using simple HTML/text since plotly not available
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Charge Type Breakdown")
        breakdown = summary["charge_type_breakdown"]
        if breakdown:
            bd_df = pd.DataFrame.from_dict(breakdown, orient="index",
                                            columns=["Count"]).sort_values("Count", ascending=False)
            bd_df.index = [i.replace("_", " ").title() for i in bd_df.index]
            st.bar_chart(bd_df)
        else:
            st.info("No hidden charges detected")

    with col_right:
        st.markdown("#### Monthly Trend")
        if not summary["monthly_trend"].empty:
            mt = summary["monthly_trend"].set_index("month")["count"]
            st.bar_chart(mt)
        else:
            st.info("Insufficient date data for trend analysis")

    st.divider()
    st.markdown("#### Risk Distribution")
    flagged = result_df[result_df["predicted_hidden_charge"] == 1]

    risk_counts = result_df["risk_level"].value_counts()
    rc_df = pd.DataFrame({"Risk Level": risk_counts.index, "Count": risk_counts.values})
    st.dataframe(rc_df, use_container_width=True, hide_index=True)

    st.markdown("#### 🚩 Flagged Transactions")
    if len(flagged) > 0:
        display_cols = ["timestamp", "description", "amount", "balance",
                        "risk_level", "predicted_charge_type", "combined_score"]
        disp = flagged[[c for c in display_cols if c in flagged.columns]].copy()
        disp["amount"] = disp["amount"].apply(lambda x: f"₹{x:,.2f}")
        disp["combined_score"] = disp["combined_score"].apply(lambda x: f"{x:.0%}")
        disp.columns = [c.replace("_", " ").title() for c in disp.columns]
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Download
        csv_data = flagged.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Flagged Transactions (CSV)",
            data=csv_data,
            file_name=f"hidden_charges_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.success("🎉 No hidden charges detected !")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: Dispute Letter Generator
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📄 Dispute Letter Generator":
    st.markdown("## 📄 Dispute Letter Generator")
    st.info("Generate a formal dispute letter to submit to your bank for hidden charges.")

    if st.session_state.results is None:
        st.warning("Please run analysis first.")
        st.stop()

    result_df = st.session_state.results
    flagged   = result_df[result_df["predicted_hidden_charge"] == 1]

    if len(flagged) == 0:
        st.success("No hidden charges to dispute!")
        st.stop()

    with st.form("dispute_form"):
        col1, col2 = st.columns(2)
        with col1:
            account_holder = st.text_input("Account Holder Name", "Your Name")
            account_number = st.text_input("Account Number (last 4 digits)", "XXXX")
        with col2:
            bank_name = st.selectbox("Bank Name", [
                "State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank",
                "Kotak Mahindra Bank", "Punjab National Bank", "Bank of Baroda",
                "Canara Bank", "Union Bank of India", "IDFC First Bank", "Other"
            ])

        st.markdown("**Select charges to dispute:**")
        charge_options = []
        for _, row in flagged.iterrows():
            desc  = str(row.get("description", ""))[:50]
            amt   = abs(float(row.get("amount", 0)))
            label = f"₹{amt:.2f} — {desc}"
            charge_options.append(label)

        selected_indices = st.multiselect(
            "Charges to include", charge_options, default=charge_options
        )
        submitted = st.form_submit_button("📝 Generate Letter")

    if submitted:
        selected_mask = [opt in selected_indices for opt in charge_options]
        selected_rows = flagged[selected_mask]

        letter = generate_dispute_letter(
            selected_rows,
            account_holder=account_holder,
            account_number=f"XXXXXXXX{account_number}",
            bank_name=bank_name,
        )

        st.markdown("#### Generated Dispute Letter")
        st.text_area("", letter, height=500)
        st.download_button(
            "📥 Download Letter (.txt)",
            data=letter.encode("utf-8"),
            file_name=f"dispute_letter_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
        )
        st.caption("⚠️ Review and customize the letter before sending. "
                   "Verify all charges independently before filing a formal complaint.")

