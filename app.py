import streamlit as st
import pandas as pd
from validator import run_validation
from chatbot import ask_chatbot, wants_visualization
from report_pdf import build_pdf_report

st.set_page_config(page_title="ValidatIQ — CSV Validator", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main { padding-top: 1.5rem; }
.stApp { background: linear-gradient(180deg, #f0f6ff 0%, #ffffff 250px); }
.stMetric { background: #e8f1ff; border: 1px solid #c7ddff; border-radius: 12px; padding: 12px; }
[data-testid="stMetricValue"] { color: #1d4ed8; }
h1 { font-weight: 800; color: #1e3a8a; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { background: #eef4ff; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: white !important; }
[data-testid="stFileUploaderDropzone"] { border: 2px dashed #3b82f6; background: #f5f9ff; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🛡️ ValidatIQ")
st.caption("Real-time CSV validation — structural checks, rule checks, and ML-based anomaly detection.")

# API key comes from YOUR secrets (local .streamlit/secrets.toml, or Streamlit Cloud's Secrets panel).
# The end user (analyst/professor) never sees or enters a key.
groq_api_key = st.secrets.get("GROQ_API_KEY", None)

uploaded_file = st.file_uploader("Drop your CSV here", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Running validation pipeline..."):
        report = run_validation(uploaded_file)

    if "error" in report:
        st.error(f"❌ {report['error']}")
    else:
        col_success, col_download = st.columns([4, 1])
        col_success.success("✅ Validation complete")
        pdf_bytes = build_pdf_report(report)
        col_download.download_button("📄 Download PDF report", data=pdf_bytes, file_name="validation_report.pdf", mime="application/pdf")

        stats = report["statistics"]
        rule_checks = report["rule_checks"]
        anomaly = report["anomaly_detection"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", stats["row_count"])
        c2.metric("Duplicate rows", rule_checks.get("_duplicate_rows", 0))
        c3.metric("Anomalies flagged", anomaly.get("flagged_count", 0))
        c4.metric("Anomaly %", f"{anomaly.get('flagged_pct', 0)}%")

        if stats.get("row_count_warning"):
            st.warning("⚠️ Row count looks unusually low compared to a typical file.")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧱 Structure & Rules", "🔤 Pattern Checks", "📊 Statistics", "🤖 ML Anomalies", "📈 Suggested KPIs"])

        with tab1:
            st.subheader("Column-level rule checks")
            rule_df = pd.DataFrame({k: v for k, v in rule_checks.items() if k != "_duplicate_rows"}).T
            st.dataframe(rule_df, use_container_width=True)

        with tab2:
            st.subheader("Whitespace & format checks (text columns)")
            if report["pattern_checks"]:
                st.dataframe(pd.DataFrame(report["pattern_checks"]).T, use_container_width=True)
            else:
                st.info("No categorical/text columns found to check.")

        with tab3:
            st.subheader("Distribution stats (numeric columns)")
            dist = stats.get("distributions", {})
            if dist:
                st.dataframe(pd.DataFrame(dist).T, use_container_width=True)
            else:
                st.info("No numeric columns found.")

        with tab4:
            st.subheader("PCA view — normal vs. anomalous rows (2D projection)")
            pca_points = anomaly.get("pca_points", [])
            if pca_points:
                pca_df = pd.DataFrame(pca_points)
                pca_df["label"] = pca_df["is_anomaly"].map({True: "Anomaly", False: "Normal"})
                st.scatter_chart(pca_df, x="x", y="y", color="label", size=40)
                st.caption("All numeric columns compressed to 2D via PCA, purely for visualization — the actual anomaly decision uses the full data.")

            st.subheader("Isolation Forest — flagged rows (most anomalous first)")
            if anomaly.get("n_clusters_found", 0) > 0:
                st.caption(f"K-Means grouped these into {anomaly['n_clusters_found']} anomaly clusters — see the 'cluster' column.")
            if anomaly.get("flagged_rows"):
                st.dataframe(pd.DataFrame(anomaly["flagged_rows"]), use_container_width=True)
            else:
                st.info(anomaly.get("message", "No anomalies detected."))

        with tab5:
            st.subheader("KPIs an analyst or engineer could build from this data")
            for kpi in report.get("kpi_suggestions", []):
                st.markdown(f"- {kpi}")

        st.divider()
        st.subheader("💬 Ask about this report")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_question = st.chat_input("Ask a question about this validation report...")

        if user_question:
            if not groq_api_key:
                st.error("⚠️ Chatbot isn't configured yet — add GROQ_API_KEY to secrets.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.write(user_question)

                with st.chat_message("assistant"):
                    if wants_visualization(user_question):
                        st.write("Here's a real chart built from your report's actual numbers (not an LLM description):")
                        null_data = {col: v.get("null_pct", 0) for col, v in report["rule_checks"].items() if col != "_duplicate_rows"}
                        if null_data:
                            st.bar_chart(null_data)
                            st.caption("Null percentage by column")
                        answer = "(Chart rendered above, generated directly from the validation report — not from the chatbot model.)"
                    else:
                        with st.spinner("Thinking..."):
                            try:
                                answer = ask_chatbot(
                                    user_question,
                                    report,
                                    groq_api_key,
                                    chat_history=st.session_state.chat_history[:-1],
                                )
                            except Exception as e:
                                answer = f"Error calling the chatbot API: {e}"
                        st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload a CSV to begin — works with any dataset, any domain.")