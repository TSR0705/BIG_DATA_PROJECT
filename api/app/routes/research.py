"""Research metrics router."""

from typing import Any, Dict
from fastapi import APIRouter
from api.app.schemas.research import ResearchMetricsResponse
from api.app.services.data_service import get_research_metrics

router = APIRouter(prefix="/api/research", tags=["Research"])


@router.get("/metrics", response_model=ResearchMetricsResponse)
def get_metrics() -> ResearchMetricsResponse:
    """Return verified validation and holdout test benchmark metrics, plus subgroup evaluations."""
    return ResearchMetricsResponse(**get_research_metrics())
