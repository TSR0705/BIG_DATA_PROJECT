"""Applicant prediction and risk schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RiskFactorImpact(BaseModel):
    name: str
    impact: float
    direction: str  # "+" or "-"


class PredictRiskRequest(BaseModel):
    age: int = Field(
        32,
        ge=18,
        le=100,
        description="Applicant age in years, must be between 18 and 100",
    )
    annual_income: float = Field(
        850000.0,
        gt=0.0,
        description="Gross annual income in INR (must be strictly positive)",
    )
    employment_type: str = Field(
        "Salaried",
        min_length=1,
        description="Employment category (e.g. Salaried, Commercial associate, Working)",
    )
    loan_amount: float = Field(
        1200000.0,
        gt=0.0,
        description="Requested loan amount in INR (must be strictly positive)",
    )
    loan_tenure_months: int = Field(
        60,
        gt=0,
        le=360,
        description="Loan repayment tenure in months (must be between 1 and 360)",
    )
    existing_obligations: float = Field(
        18000.0,
        ge=0.0,
        description="Existing monthly debt obligations in INR (must be non-negative)",
    )


class PredictRiskResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    applicant_id: str
    default_probability: float
    default_percentage: float
    risk_tier: str  # "Low Risk", "Medium Risk", "High Risk"
    risk_category: str  # "Low", "Medium", "High"
    confidence: str  # "High" or "Demonstration Only"
    model_used: str  # "Illustrative Heuristic Estimate" or "XGBoost (Calibrated - 256 Features)"
    decision: str  # "APPROVE" or "REJECT"
    summary: str
    risk_factors: List[RiskFactorImpact]
    inference_type: str = "ILLUSTRATIVE_HEURISTIC"  # "GENUINE_MODEL" or "ILLUSTRATIVE_HEURISTIC"
    explanation_type: str = "ILLUSTRATIVE_HEURISTIC_FACTORS"  # "GENUINE_TREESHAP" or "ILLUSTRATIVE_HEURISTIC_FACTORS"
    disclaimer: Optional[str] = None
