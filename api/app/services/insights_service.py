"""Dataset and research insights computed from committed artifacts (never hard-coded).

Sources
-------
- reports/split_report.json ............ applicant and default counts (Phase 6)
- reports/subgroup_analysis.csv ........ validation metrics of the calibrated model (Phase 11, 'Overall' row)
- artifacts/models/phase11_calibration_metadata.json ... calibration method
- artifacts/explainability/test_split_shap_values.parquet ... SHAP matrix of the frozen test split (Phase 11)
- data/model_input/model_input.parquet .. labelled model-ready table (Phase 5) for default-rate profiles

Expensive computations (SHAP ranking, default-rate profiles) are computed once and
cached as JSON under artifacts/dashboard_cache/ with the source file's mtime, so the
cache is rebuilt automatically when an artifact changes.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from api.app.core.config import (
    DASHBOARD_CACHE_DIR,
    MODEL_INPUT_PATH,
    OPERATING_THRESHOLD,
    PHASE11_CALIBRATION_METADATA_PATH,
    SHAP_TEST_VALUES_PATH,
    SPLIT_REPORT_PATH,
    SUBGROUP_ANALYSIS_PATH,
)
from api.app.core.feature_labels import feature_label

ID_COL = "SK_ID_CURR"


# --------------------------------------------------------------------------- cache helpers
def _mtime(path) -> float:
    if path.is_dir():
        return max(os.path.getmtime(p) for p in path.iterdir()) if any(path.iterdir()) else 0.0
    return os.path.getmtime(path) if path.exists() else 0.0


def _cached(name: str, sources: List, builder):
    DASHBOARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DASHBOARD_CACHE_DIR / f"{name}.json"
    stamp = max(_mtime(p) for p in sources)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("_source_mtime") == stamp:
            return payload["data"]
    data = builder()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"_source_mtime": stamp, "data": data}, f)
    return data


# --------------------------------------------------------------------------- headline metrics
def dataset_counts() -> Dict[str, Any]:
    """Applicant/default counts from the Phase 6 split report."""
    with open(SPLIT_REPORT_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)
    total = rep["splits"]["total"] if "splits" in rep else rep["total"]
    rows = int(total["rows"])
    positives = int(total["positive_count"])
    return {
        "total_applicants": rows,
        "total_defaults": positives,
        "non_defaults": rows - positives,
        "default_rate": positives / rows,
        "source": "reports/split_report.json",
    }


def validation_metrics() -> Dict[str, Any]:
    """Validation metrics of the deployed (Phase 8 XGBoost + Phase 11 isotonic) model."""
    df = pd.read_csv(SUBGROUP_ANALYSIS_PATH)
    row = df[(df["subgroup_dimension"] == "Overall")].iloc[0]
    with open(PHASE11_CALIBRATION_METADATA_PATH, "r", encoding="utf-8") as f:
        cal = json.load(f)
    return {
        "partition": "Validation",
        "sample_size": int(row["sample_size"]),
        "roc_auc": float(row["roc_auc"]),
        "roc_auc_ci": str(row["roc_auc_ci"]),
        "pr_auc": float(row["pr_auc"]),
        "brier_score": float(row["brier_score"]),
        "ece": float(row["ece"]),
        "ece_ci": str(row["ece_ci"]),
        "precision_at_threshold": float(row["precision_at_016"]),
        "recall_at_threshold": float(row["recall_at_016"]),
        "operating_threshold": OPERATING_THRESHOLD,
        "calibration_method": cal.get("calibration_method", "isotonic"),
        "model_version": "XGBoost Tuned (Phase 8) + Isotonic OOF Calibrator (Phase 11)",
        "source": "reports/subgroup_analysis.csv (Overall row)",
    }


def subgroup_roc_auc_series() -> List[Dict[str, Any]]:
    """ROC-AUC of the deployed model per evaluated subgroup (Phase 11), excluding the overall row."""
    df = pd.read_csv(SUBGROUP_ANALYSIS_PATH)
    df = df[df["subgroup_dimension"] != "Overall"]
    return [
        {"name": str(r["subgroup_name"]), "roc_auc": float(r["roc_auc"])}
        for _, r in df.iterrows() if pd.notna(r["roc_auc"])
    ]


# --------------------------------------------------------------------------- default-rate profile
PROFILE_DIMENSIONS = {
    "age": {
        "label": "Applicant age",
        "column": "DAYS_BIRTH",
        "transform": lambda s: (-s) / 365.25,
        "bins": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "fmt": lambda lo, hi: f"{int(lo)}–{int(hi)}",
        "axis": "Age band (years)",
    },
    "credit": {
        "label": "Loan amount",
        "column": "AMT_CREDIT",
        "transform": lambda s: s / 1000.0,
        "bins": [0, 150, 300, 450, 600, 800, 1000, 1300, 1700, 2500, 5000],
        "fmt": lambda lo, hi: f"{int(lo)}–{int(hi)}k",
        "axis": "Loan amount band (thousand currency units)",
    },
    "income": {
        "label": "Annual income",
        "column": "AMT_INCOME_TOTAL",
        "transform": lambda s: s / 1000.0,
        "bins": [0, 75, 100, 125, 150, 180, 225, 300, 400, 600, 1000],
        "fmt": lambda lo, hi: f"{int(lo)}–{int(hi)}k",
        "axis": "Income band (thousand currency units)",
    },
}


def _build_profile(dimension: str) -> Dict[str, Any]:
    spec = PROFILE_DIMENSIONS[dimension]
    df = pq.read_table(MODEL_INPUT_PATH, columns=["TARGET", spec["column"]]).to_pandas()
    values = spec["transform"](df[spec["column"]].astype("float64"))
    bins = spec["bins"]
    cats = pd.cut(values, bins=bins, right=False, include_lowest=True)
    grouped = df.groupby(cats, observed=False)["TARGET"].agg(["mean", "count"])
    bands = []
    for interval, (rate, n) in grouped.iterrows():
        if n == 0:
            continue
        bands.append({
            "band": spec["fmt"](interval.left, interval.right),
            "lower": float(interval.left),
            "upper": float(interval.right),
            "applicants": int(n),
            "defaults": int(round(rate * n)),
            "default_rate": round(float(rate) * 100, 2),
        })
    outside = int(len(df) - sum(b["applicants"] for b in bands))
    peak = max(bands, key=lambda b: b["default_rate"]) if bands else None
    return {
        "dimension": dimension,
        "label": spec["label"],
        "axis_title": spec["axis"],
        "feature": spec["column"],
        "overall_default_rate": round(float(df["TARGET"].mean()) * 100, 2),
        "bands": bands,
        "peak_band": peak["band"] if peak else None,
        "applicants_outside_bands": outside,
        "source": "data/model_input/model_input.parquet (TARGET by band)",
    }


def default_rate_profile(dimension: str = "age") -> Dict[str, Any]:
    if dimension not in PROFILE_DIMENSIONS:
        raise KeyError(dimension)
    return _cached(f"default_rate_profile_{dimension}", [MODEL_INPUT_PATH], lambda: _build_profile(dimension))


# --------------------------------------------------------------------------- SHAP risk factors
def _build_risk_factors() -> Dict[str, Any]:
    shap = pq.read_table(SHAP_TEST_VALUES_PATH).to_pandas()
    feature_cols = [c for c in shap.columns if c != ID_COL]
    mean_abs = shap[feature_cols].abs().mean().sort_values(ascending=False)
    top = list(mean_abs.index[:20])
    ids = shap[ID_COL].tolist()
    feats = pq.read_table(MODEL_INPUT_PATH, columns=[ID_COL] + top, filters=[(ID_COL, "in", ids)]).to_pandas()
    merged = shap[[ID_COL] + top].merge(feats, on=ID_COL, suffixes=("_shap", "_val"))
    factors = []
    for rank, f in enumerate(top, start=1):
        val = merged[f"{f}_val"]
        sv = merged[f"{f}_shap"].astype("float64")
        is_categorical = val.dtype == object or str(val.dtype).startswith("category")
        if is_categorical:
            direction = "categorical"
            corr = None
        else:
            v = val.astype("float64")
            mask = v.notna()
            corr = float(np.corrcoef(v[mask].rank(), sv[mask].rank())[0, 1]) if mask.sum() > 10 else 0.0
            direction = "increase" if corr > 0 else "decrease"
        factors.append({
            "rank": rank,
            "feature": f,
            "display_name": feature_label(f),
            "mean_abs_shap": round(float(mean_abs[f]), 4),
            "mean_shap": round(float(sv.mean()), 4),
            "direction": direction,
            "rank_correlation": None if corr is None else round(corr, 3),
        })
    return {
        "sample_size": int(len(shap)),
        "partition": "Frozen test split",
        "units": "log-odds",
        "factors": factors,
        "source": "artifacts/explainability/test_split_shap_values.parquet",
    }


def top_risk_factors(k: int = 5) -> Dict[str, Any]:
    data = _cached("shap_risk_factors", [SHAP_TEST_VALUES_PATH], _build_risk_factors)
    out = dict(data)
    out["factors"] = data["factors"][:k]
    return out
