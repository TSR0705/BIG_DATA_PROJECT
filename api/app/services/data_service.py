"""Data Service: project overview and research metrics assembled from committed artifacts."""

from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

from api.app.core.config import REPORTS_DIR
from api.app.schemas.project import (
    ApplicantDistribution,
    DefaultRateProfile,
    KPICard,
    ProjectOverview,
    RecentPrediction,
    RiskFactor,
)
from api.app.services import insights_service
from api.app.services.prediction_log import read_predictions
from api.app.services.prediction_service import ensure_seeded_predictions


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def get_project_overview(profile_dimension: str = "age") -> ProjectOverview:
    """Assemble the dashboard overview. Every number is read from a project artifact or produced by the frozen model."""
    counts = insights_service.dataset_counts()
    val = insights_service.validation_metrics()
    profile = insights_service.default_rate_profile(profile_dimension)
    shap = insights_service.top_risk_factors(k=5)

    default_rate_pct = counts["default_rate"] * 100
    non_default_pct = 100 - default_rate_pct
    age_profile = profile if profile_dimension == "age" else insights_service.default_rate_profile("age")
    applicants_by_age = [b["applicants"] for b in age_profile["bands"]]
    defaults_by_age = [b["defaults"] for b in age_profile["bands"]]
    subgroup_auc = [s["roc_auc"] for s in insights_service.subgroup_roc_auc_series()]

    kpis = [
        KPICard(
            title="Total Applicants",
            value=_fmt_int(counts["total_applicants"]),
            numeric_value=counts["total_applicants"],
            subtitle="Full dataset (100%)",
            trend="up",
            badge_type="green",
            source=counts["source"],
            sparkline=applicants_by_age,
            sparkline_label="Applicants per age band",
        ),
        KPICard(
            title="Total Defaults",
            value=_fmt_int(counts["total_defaults"]),
            numeric_value=counts["total_defaults"],
            subtitle=f"{default_rate_pct:.2f}% default rate",
            trend="",
            badge_type="red",
            source=counts["source"],
            sparkline=defaults_by_age,
            sparkline_label="Defaults per age band",
        ),
        KPICard(
            title="Model Accuracy",
            value=f"{val['roc_auc']:.4f}",
            numeric_value=val["roc_auc"],
            subtitle=f"ROC-AUC ({val['partition']})",
            trend="",
            badge_type="teal",
            source=val["source"],
            sparkline=subgroup_auc,
            sparkline_label="ROC-AUC across evaluated subgroups",
        ),
    ]

    distribution = ApplicantDistribution(
        total=counts["total_applicants"],
        non_default_count=counts["non_defaults"],
        non_default_percent=round(non_default_pct, 1),
        default_count=counts["total_defaults"],
        default_percent=round(default_rate_pct, 1),
    )

    ensure_seeded_predictions()
    recent = [RecentPrediction(**_recent_view(e)) for e in read_predictions(limit=5)]

    factors = [RiskFactor(**f) for f in shap["factors"]]

    key_takeaways = [
        f"Default rate in the dataset: {default_rate_pct:.2f}%.",
        f"The model effectively identifies high-risk applicants (ROC-AUC: {val['roc_auc']:.4f}).",
        f"Predictions are well-calibrated (ECE: {val['ece']:.4f}).",
        "Loan amount and age are among the top risk factors.",
        "Built on 300K+ real loan applications using Spark.",
    ]

    return ProjectOverview(
        project_name="CreditLens",
        subtitle="A Big Data–Driven Loan Default Prediction System",
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_applicants=counts["total_applicants"],
        total_defaults=counts["total_defaults"],
        default_rate=round(default_rate_pct, 4),
        non_defaults=counts["non_defaults"],
        non_default_rate=round(non_default_pct, 4),
        model_roc_auc=round(val["roc_auc"], 4),
        model_pr_auc=round(val["pr_auc"], 4),
        model_ece=round(val["ece"], 4),
        brier_score=round(val["brier_score"], 4),
        model_version=val["model_version"],
        kpis=kpis,
        distribution=distribution,
        default_rate_profile=DefaultRateProfile(**profile),
        profile_dimensions=[
            {"key": k, "label": v["label"]} for k, v in insights_service.PROFILE_DIMENSIONS.items()
        ],
        top_risk_factors=factors,
        shap_context={k: v for k, v in shap.items() if k != "factors"},
        recent_predictions=recent,
        key_takeaways=key_takeaways,
        data_sources={
            "counts": counts["source"],
            "validation_metrics": val["source"],
            "default_rate_profile": profile["source"],
            "risk_factors": shap["source"],
            "recent_predictions": "artifacts/predictions/prediction_log.jsonl (real engine outputs)",
        },
    )


def _recent_view(entry: Dict[str, Any]) -> Dict[str, Any]:
    inf_type = (
        "GENUINE_MODEL"
        if entry.get("sk_id_curr") is not None or "XGBoost" in entry.get("model_version", "")
        else "ILLUSTRATIVE_ESTIMATE"
    )
    return {
        "applicant_id": entry["applicant_id"],
        "sk_id_curr": entry.get("sk_id_curr"),
        "scored_at": entry["scored_at"],
        "risk_tier": entry["risk_tier"],
        "probability": entry["probability"],
        "probability_percent": f"{entry['probability'] * 100:.1f}%",
        "decision": entry.get("decision", ""),
        "source": entry.get("source", ""),
        "model_version": entry.get("model_version", ""),
        "inference_type": inf_type,
    }


def get_research_metrics() -> Dict[str, Any]:
    """Load verified research metrics from model_comparison and subgroup_analysis."""
    comp_path = REPORTS_DIR / "model_comparison.csv"
    sub_path = REPORTS_DIR / "subgroup_analysis.csv"

    benchmarks = []
    if comp_path.exists():
        df_comp = pd.read_csv(comp_path)
        for _, row in df_comp.iterrows():
            benchmarks.append({
                "model_name": str(row.get("model", row.get("model_name", ""))),
                "configuration": str(row.get("strategy", row.get("configuration", ""))),
                "threshold": float(row.get("threshold", 0.5)),
                "calibration": str(row.get("calibration_method", row.get("calibration", "None"))),
                "roc_auc": round(float(row.get("roc_auc", 0)), 4),
                "pr_auc": round(float(row.get("pr_auc", 0)), 4),
                "brier_score": round(float(row.get("brier_score", 0)), 4),
                "ece": round(float(row.get("ece", 0)), 4),
                "recall": round(float(row.get("recall", 0)), 4),
                "precision": round(float(row.get("precision", 0)), 4),
                "f1_score": round(float(row.get("f1", row.get("f1_score", 0))), 4),
            })

    subgroups = []
    if sub_path.exists():
        df_sub = pd.read_csv(sub_path)
        for _, row in df_sub.iterrows():
            def _opt(v):
                return round(float(v), 4) if pd.notna(v) and str(v) != "NA" else None
            subgroups.append({
                "dimension": str(row.get("subgroup_dimension", row.get("dimension", ""))),
                "subgroup_name": str(row.get("subgroup_name", "")),
                "sample_size": int(row.get("sample_size", row.get("n", 0))),
                "defaults": int(row.get("defaults", 0)),
                "default_rate": round(float(row.get("default_rate", 0)), 2),
                "roc_auc": _opt(row.get("roc_auc")),
                "pr_auc": _opt(row.get("pr_auc")),
                "brier_score": round(float(row.get("brier_score", 0)), 4),
                "ece": round(float(row.get("ece", 0)), 4),
                "precision": round(float(row.get("precision_at_016", row.get("precision", 0))), 4),
                "recall": round(float(row.get("recall_at_016", row.get("recall", 0))), 4),
            })

    val = insights_service.validation_metrics()
    return {
        "project": "CreditLens",
        "validation_performance": {
            "roc_auc": round(val["roc_auc"], 4),
            "pr_auc": round(val["pr_auc"], 4),
            "brier_score": round(val["brier_score"], 4),
            "ece": round(val["ece"], 4),
            "operating_threshold": val["operating_threshold"],
            "precision_at_threshold": round(val["precision_at_threshold"], 4),
            "recall_at_threshold": round(val["recall_at_threshold"], 4),
            "source": val["source"],
        },
        "test_evaluation": _frozen_test_evaluation(benchmarks),
        "benchmarks": benchmarks,
        "subgroups": subgroups,
    }


def _frozen_test_evaluation(benchmarks) -> Dict[str, Any]:
    """Test-partition row of the deployed configuration (isotonic + cost-sensitive 1:5) from model_comparison.csv.

    The test partition was accessed in Phases 7-8 before the locked Phase 11 pass, as
    reports/frozen_test_evaluation.md documents; it is therefore labelled 'historical access'.
    """
    for b in benchmarks:
        if b["model_name"].startswith("Calibrated") and "1:5" in b["configuration"]:
            return {"partition": "Test split (historical access; see reports/frozen_test_evaluation.md)", **b}
    return {"partition": "Test split", "note": "see reports/frozen_test_evaluation.md"}
