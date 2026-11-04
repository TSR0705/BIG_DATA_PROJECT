"""Model metadata and feature routes."""

from typing import Any, Dict, List
from fastapi import APIRouter
from api.app.schemas.model import ModelMetadataResponse
from api.app.services.model_service import get_model_metadata, get_inference_engine

router = APIRouter(prefix="/api/model", tags=["Model"])


@router.get("/metadata", response_model=ModelMetadataResponse)
def get_metadata() -> ModelMetadataResponse:
    """Return model parameters, architecture, calibration, and threshold specifications."""
    return ModelMetadataResponse(**get_model_metadata())


@router.get("/features")
def get_features() -> Dict[str, Any]:
    """Return list of all 256 required model features with type classification."""
    engine = get_inference_engine()
    cat_set = set(engine.categorical_columns)
    features = [
        {
            "name": name,
            "type": "categorical" if name in cat_set else "numeric",
        }
        for name in engine.feature_names
    ]
    return {
        "total_features": len(features),
        "categorical_count": len(cat_set),
        "numeric_count": len(features) - len(cat_set),
        "features": features,
    }
