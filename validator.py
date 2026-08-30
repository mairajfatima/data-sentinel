"""
Core data validation engine.
Three parts: type inference (engineering), rule checks (engineering),
anomaly detection (machine learning - Isolation Forest, unsupervised).
"""

import re
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_file(file) -> dict:
    """Phase 1: is this even a readable, non-empty CSV before we do anything else?"""
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return {"valid": False, "error": f"Could not read file as CSV: {e}"}
    if df.empty:
        return {"valid": False, "error": "File is empty — no rows found."}
    if len(df.columns) == 0:
        return {"valid": False, "error": "No columns detected."}
    file.seek(0)  # reset pointer so it can be read again downstream
    return {"valid": True, "row_count": len(df), "column_count": len(df.columns)}


def load_and_infer(file) -> tuple[pd.DataFrame, dict]:
    """Load a CSV and classify each column: numeric, datetime, or categorical."""
    df = pd.read_csv(file)
    roles = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            roles[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            roles[col] = "datetime"
        else:
            try:
                pd.to_datetime(df[col], errors="raise")
                roles[col] = "datetime"
            except Exception:
                roles[col] = "categorical"
    return df, roles


def check_rules(df: pd.DataFrame, roles: dict) -> dict:
    """Rule-based checks: nulls, duplicates, negative numbers where unexpected."""
    results = {}
    for col, role in roles.items():
        col_result = {
            "nulls": int(df[col].isnull().sum()),
            "null_pct": round(float(df[col].isnull().mean()) * 100, 2),
        }
        if role == "numeric":
            col_result["negative_count"] = int((df[col] < 0).sum())
            col_result["min"] = float(df[col].min()) if df[col].notna().any() else None
            col_result["max"] = float(df[col].max()) if df[col].notna().any() else None
        if role == "categorical":
            col_result["unique_values"] = int(df[col].nunique())
        results[col] = col_result
    results["_duplicate_rows"] = int(df.duplicated().sum())
    return results


def check_patterns(df: pd.DataFrame, roles: dict) -> dict:
    """Phase 2: whitespace issues + basic email-format check on text columns."""
    results = {}
    for col, role in roles.items():
        if role != "categorical":
            continue
        series = df[col].dropna().astype(str)
        has_whitespace = int((series != series.str.strip()).sum())
        col_result = {"leading_trailing_whitespace": has_whitespace}
        looks_like_email_col = "email" in col.lower()
        if looks_like_email_col:
            invalid_emails = int((~series.str.match(EMAIL_PATTERN)).sum())
            col_result["invalid_email_format"] = invalid_emails
        results[col] = col_result
    return results


def check_statistics(df: pd.DataFrame, roles: dict, expected_min_rows: int = 10) -> dict:
    """Phase 4: distribution stats + row-count sanity check."""
    row_count = len(df)
    stats = {
        "row_count": row_count,
        "row_count_warning": row_count < expected_min_rows,
    }
    distributions = {}
    for col, role in roles.items():
        if role == "numeric":
            distributions[col] = {
                "mean": round(float(df[col].mean()), 4) if df[col].notna().any() else None,
                "median": round(float(df[col].median()), 4) if df[col].notna().any() else None,
                "std": round(float(df[col].std()), 4) if df[col].notna().any() else None,
            }
    stats["distributions"] = distributions
    return stats


def explain_anomaly(row_values: dict, column_stats: dict, top_n: int = 2) -> str:
    """
    Derive a human-readable reason for why a row was flagged, by ranking
    which numeric columns deviate most from their column's mean (in std units).
    Binary/near-constant columns (like a 0/1 flag) are excluded — their low
    variance makes any "1" look like a huge deviation even though it carries
    little information, which would drown out genuinely informative columns.
    """
    deviations = []
    for col, value in row_values.items():
        stat = column_stats.get(col)
        if not stat or stat.get("std") in (None, 0):
            continue
        if stat.get("is_near_binary"):
            continue
        z = abs((value - stat["mean"]) / stat["std"])
        deviations.append((col, value, z))

    if not deviations:
        return "Flagged as an unusual combination of values across columns (including flag/category columns)."

    deviations.sort(key=lambda d: d[2], reverse=True)
    top = deviations[:top_n]
    parts = []
    for col, value, z in top:
        mean = column_stats[col]["mean"]
        direction = "high" if value > mean else "low"
        parts.append(f"{col}={value:.2f} (unusually {direction} vs. average, {z:.1f} std devs away)")
    return "Driven mainly by: " + "; ".join(parts)


def detect_anomalies(df: pd.DataFrame, roles: dict, contamination: float = 0.05, n_clusters: int = 3) -> dict:
    """
    ML step, now three algorithms working together:
    1. StandardScaler - put all numeric columns on the same scale (required before PCA/clustering,
       otherwise a column like 'salary' dominates a column like 'age' purely due to bigger numbers).
    2. PCA - reduce all numeric columns to 2 dimensions, purely for visualization.
    3. Isolation Forest - the actual anomaly detector, run on the SCALED ORIGINAL columns
       (not the 2D PCA version - PCA loses information, so we keep full data for the real decision).
    4. K-Means - cluster only the flagged anomalies into groups of "similar unusual rows".
    """
    numeric_cols = [c for c, r in roles.items() if r == "numeric"]
    if len(numeric_cols) < 1:
        return {"message": "No numeric columns available for anomaly detection.", "flagged_rows": [], "pca_points": []}

    numeric_df = df[numeric_cols].fillna(df[numeric_cols].mean())
    column_stats = {
        col: {
            "mean": float(numeric_df[col].mean()),
            "std": float(numeric_df[col].std()),
            "is_near_binary": numeric_df[col].nunique() <= 2,
        }
        for col in numeric_cols
    }

    # 1. Scale - mean 0, std 1 for every column, so no single column dominates by magnitude alone
    scaled = StandardScaler().fit_transform(numeric_df)

    # 2. PCA - 2D projection purely for visualization (not used for the anomaly decision itself)
    n_components = min(2, scaled.shape[1])
    pca_coords = PCA(n_components=n_components, random_state=42).fit_transform(scaled)

    # 3. Isolation Forest - the real anomaly decision, on the full scaled feature set
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(scaled)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(scaled)

    is_anomaly = predictions == -1

    # 4. K-Means - group the flagged anomalies into sub-types (only runs if enough anomalies exist)
    cluster_labels = {}
    if is_anomaly.sum() >= n_clusters:
        anomaly_scaled = scaled[is_anomaly]
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(anomaly_scaled)
        anomaly_indices = df.index[is_anomaly]
        cluster_labels = dict(zip(anomaly_indices, labels))

    flagged_rows = []
    for idx in df.index[is_anomaly]:
        row_values = df.loc[idx, numeric_cols].to_dict()
        flagged_rows.append({
            "row_index": int(idx),
            "anomaly_score": round(float(scores[df.index.get_loc(idx)]), 4),
            "cluster": int(cluster_labels.get(idx, -1)) if cluster_labels else None,
            "values": row_values,
            "reason": explain_anomaly(row_values, column_stats),
        })
    flagged_rows.sort(key=lambda r: r["anomaly_score"])

    pca_points = [
        {"x": round(float(pca_coords[i, 0]), 3),
         "y": round(float(pca_coords[i, 1]), 3) if n_components > 1 else 0.0,
         "is_anomaly": bool(is_anomaly[i])}
        for i in range(len(df))
    ]

    return {
        "total_rows": len(df),
        "flagged_count": int(is_anomaly.sum()),
        "flagged_pct": round(float(is_anomaly.sum()) / len(df) * 100, 2),
        "flagged_rows": flagged_rows[:20],
        "pca_points": pca_points,
        "n_clusters_found": len(set(cluster_labels.values())) if cluster_labels else 0,
    }


def suggest_kpis(df: pd.DataFrame, roles: dict) -> list[str]:
    """
    Suggest KPIs a data analyst/engineer could build from this specific dataset,
    based purely on what column types are actually present — no hardcoded domain.
    """
    numeric_cols = [c for c, r in roles.items() if r == "numeric"]
    categorical_cols = [c for c, r in roles.items() if r == "categorical"]
    datetime_cols = [c for c, r in roles.items() if r == "datetime"]

    suggestions = []

    for col in numeric_cols[:5]:
        suggestions.append(f"Total and average {col} (sum/mean across all rows)")

    for cat in categorical_cols[:3]:
        for num in numeric_cols[:2]:
            suggestions.append(f"{num} broken down by {cat} (group-by aggregation)")

    if datetime_cols:
        date_col = datetime_cols[0]
        for num in numeric_cols[:2]:
            suggestions.append(f"{num} trend over time, using {date_col}")

    for cat in categorical_cols[:3]:
        suggestions.append(f"Distribution / share of records by {cat}")

    if not suggestions:
        suggestions.append("Not enough numeric/categorical structure to suggest KPIs automatically.")

    return suggestions[:10]


def build_report(rule_result: dict, pattern_result: dict, stats_result: dict, anomaly_result: dict, kpi_suggestions: list) -> dict:
    """Combine everything into one structured report."""
    return {
        "rule_checks": rule_result,
        "pattern_checks": pattern_result,
        "statistics": stats_result,
        "anomaly_detection": anomaly_result,
        "kpi_suggestions": kpi_suggestions,
    }


def run_validation(file, contamination: float = 0.05) -> dict:
    """Main entry point — the only function the UI needs to call."""
    file_check = validate_file(file)
    if not file_check["valid"]:
        return {"error": file_check["error"]}

    df, roles = load_and_infer(file)
    rules = check_rules(df, roles)
    patterns = check_patterns(df, roles)
    stats = check_statistics(df, roles)
    anomalies = detect_anomalies(df, roles, contamination=contamination)
    kpis = suggest_kpis(df, roles)
    return build_report(rules, patterns, stats, anomalies, kpis)