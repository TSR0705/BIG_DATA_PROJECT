"""Health check router."""

from fastapi import APIRouter
from api.app.schemas.project import HealthResponse
from api.app.services.model_service import get_inference_engine

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return system and model readiness status."""
    engine = get_inference_engine()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        project="CreditLens - Loan Default Prediction",
        models_loaded=engine is not None,
        calibrator_loaded=engine._calibrator is not None,
    )
