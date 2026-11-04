"""Model metadata and feature schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class FeatureInfo(BaseModel):
    name: str
    feature_type: str  # "categorical" or "numeric"
    description: Optional[str] = None


class ModelMetadataResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    model_type: str
    version: str
    total_features: int
    categorical_features_count: int
    numeric_features_count: int
    hyperparameters: Dict[str, Any]
    calibration: Dict[str, Any]
    thresholds: Dict[str, float]
    risk_bands: Dict[str, str]
    provenance: Dict[str, Any]
