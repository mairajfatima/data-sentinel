"""Generates a downloadable PDF summary of a validation report."""

from fpdf import FPDF


def _safe(text: str) -> str:
    """Strip characters the default PDF font can't render (e.g. em-dash, sigma)."""
    return str(text).encode("latin-1", "ignore").decode("latin-1")


def build_pdf_report(report: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "ValidatIQ - Data Validation Report", ln=True)
    pdf.set_font("Helvetica", "", 11)

    stats = report.get("statistics", {})
    anomaly = report.get("anomaly_detection", {})

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Rows: {stats.get('row_count', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Duplicate rows: {report.get('rule_checks', {}).get('_duplicate_rows', 0)}", ln=True)
    pdf.cell(0, 7, f"Anomalies flagged: {anomaly.get('flagged_count', 0)} ({anomaly.get('flagged_pct', 0)}%)", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Top Anomalies", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for row in anomaly.get("flagged_rows", [])[:10]:
        text = f"Row {row['row_index']} (score {row['anomaly_score']}): {row.get('reason', '')}"
        pdf.multi_cell(0, 6, _safe(text))
        pdf.set_xy(pdf.l_margin, pdf.get_y())

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Suggested KPIs", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for kpi in report.get("kpi_suggestions", []):
        pdf.multi_cell(0, 6, _safe(f"- {kpi}"))
        pdf.set_xy(pdf.l_margin, pdf.get_y())

    return bytes(pdf.output())