"""Prediction log routes (real engine outputs only)."""

from typing import Any, Dict, List

from fastapi import APIRouter, Query

from api.app.schemas.project import RecentPrediction
from api.app.services.data_service import _recent_view
from api.app.services.prediction_log import count_predictions, read_predictions
from api.app.services.prediction_service import ensure_seeded_predictions

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])


@router.get("/recent", response_model=List[RecentPrediction])
def get_recent_predictions(limit: int = Query(5, ge=1, le=50)) -> List[RecentPrediction]:
    """Most recent predictions produced by the frozen model through this API, newest first."""
    ensure_seeded_predictions()
    return [RecentPrediction(**_recent_view(e)) for e in read_predictions(limit=limit)]


@router.get("/stats")
def get_prediction_stats() -> Dict[str, Any]:
    ensure_seeded_predictions()
    return {"logged_predictions": count_predictions()}
