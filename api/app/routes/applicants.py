"""Applicant lookup and scoring routes (frozen engine, real applicant rows)."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from api.app.schemas.applicants import PredictRiskRequest, PredictRiskResponse
from api.app.services.applicant_store import partition_of, search_ids
from api.app.services.prediction_service import (
    predict_applicant_risk,
    score_known_applicant,
    score_sample,
)

router = APIRouter(prefix="/api/applicants", tags=["Applicants"])


@router.post("/predict-risk", response_model=PredictRiskResponse)
def predict_risk(req: PredictRiskRequest) -> PredictRiskResponse:
    """Predict loan default probability, risk tier and risk factors from user-supplied inputs."""
    result = predict_applicant_risk(
        age=req.age,
        annual_income=req.annual_income,
        employment_type=req.employment_type,
        loan_amount=req.loan_amount,
        loan_tenure_months=req.loan_tenure_months,
        existing_obligations=req.existing_obligations,
    )
    return PredictRiskResponse(**result)


@router.get("/example", response_model=PredictRiskRequest)
def get_example_applicant() -> PredictRiskRequest:
    """Return benchmark example applicant details shown in the design authority."""
    return PredictRiskRequest(
        age=32,
        annual_income=850000.0,
        employment_type="Salaried",
        loan_amount=1200000.0,
        loan_tenure_months=60,
        existing_obligations=18000.0,
    )


@router.get("/search")
def search_applicants(q: str = Query(..., min_length=1, max_length=12), limit: int = Query(8, ge=1, le=20)) -> Dict[str, Any]:
    """Prefix search over applicant IDs (SK_ID_CURR) in the labelled and inference tables."""
    hits = search_ids(q, limit=limit)
    return {"query": q, "count": len(hits), "results": hits}


@router.get("/sample")
def get_sample_applicants(limit: int = Query(5, ge=1, le=20)) -> List[Dict[str, Any]]:
    """Score the first `limit` applicants of the frozen test split with the real engine (logged)."""
    try:
        return score_sample(limit=limit, source="sample")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{sk_id_curr}/score")
def score_applicant(sk_id_curr: int) -> Dict[str, Any]:
    """Score one known applicant by ID. Returns 404 if the ID is not in the model-ready tables."""
    if partition_of(sk_id_curr) is None:
        raise HTTPException(status_code=404, detail=f"Applicant {sk_id_curr} not found")
    result = score_known_applicant(sk_id_curr, source="search")
    if result is None:
        raise HTTPException(status_code=404, detail=f"Applicant {sk_id_curr} not found")
    return result
