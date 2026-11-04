"""Project overview and health schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.1.0"
    project: str = "CreditLens - Loan Default Prediction"
    models_loaded: bool = True
    calibrator_loaded: bool = True


class KPICard(BaseModel):
    title: str
    value: str
    numeric_value: float
    subtitle: str
    trend: str
    badge_type: str  # "green", "red", "teal"
    source: str
    sparkline: List[float] = []
    sparkline_label: str = ""


class RiskFactor(BaseModel):
    rank: int
    feature: str
    display_name: str
    mean_abs_shap: float
    mean_shap: float
    direction: str  # "increase" | "decrease" | "categorical"
    rank_correlation: Optional[float] = None


class RecentPrediction(BaseModel):
    applicant_id: str
    sk_id_curr: Optional[int] = None
    scored_at: str
    risk_tier: str  # "Low Risk", "Medium Risk", "High Risk"
    probability: float
    probability_percent: str
    decision: str = ""
    source: str = ""


class ProfileBand(BaseModel):
    band: str
    lower: float
    upper: float
    applicants: int
    defaults: int
    default_rate: float


class DefaultRateProfile(BaseModel):
    dimension: str
    label: str
    axis_title: str
    feature: str
    overall_default_rate: float
    bands: List[ProfileBand]
    peak_band: Optional[str] = None
    applicants_outside_bands: int = 0
    source: str


class ApplicantDistribution(BaseModel):
    total: int
    non_default_count: int
    non_default_percent: float
    default_count: int
    default_percent: float


class ProjectOverview(BaseModel):
    model_config = {"protected_namespaces": ()}

    project_name: str
    subtitle: str
    generated_at: str
    total_applicants: int
    total_defaults: int
    default_rate: float
    non_defaults: int
    non_default_rate: float
    model_roc_auc: float
    model_pr_auc: float
    model_ece: float
    brier_score: float
    model_version: str
    kpis: List[KPICard]
    distribution: ApplicantDistribution
    default_rate_profile: DefaultRateProfile
    profile_dimensions: List[Dict[str, str]]
    top_risk_factors: List[RiskFactor]
    shap_context: Dict[str, Any]
    recent_predictions: List[RecentPrediction]
    key_takeaways: List[str]
    data_sources: Dict[str, str]
