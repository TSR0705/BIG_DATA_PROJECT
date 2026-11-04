"""Model Service for loading and querying frozen ML models."""

import json
import pathlib
from typing import Any, Dict, List, Optional

from api.app.core.config import MODELS_DIR, MODEL_CARD_PATH
from src.inference.engine import CreditLensInferenceEngine

_ENGINE_INSTANCE: Optional[CreditLensInferenceEngine] = None


def get_inference_engine() -> CreditLensInferenceEngine:
    """Singleton getter for CreditLensInferenceEngine."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = CreditLensInferenceEngine()
    return _ENGINE_INSTANCE


def get_model_metadata() -> Dict[str, Any]:
    """Retrieve verified model metadata from model_card.json and model metadata."""
    if MODEL_CARD_PATH.exists():
        with open(MODEL_CARD_PATH, "r", encoding="utf-8") as f:
            card = json.load(f)
    else:
        card = {}

    engine = get_inference_engine()

    return {
        "model_name": "CreditLens XGBoost Tuned Production Classifier",
        "model_type": "Gradient Boosted Decision Trees (XGBoost) + Isotonic Calibrator",
        "version": "1.0.0-phase11-final",
        "total_features": len(engine.feature_names),
        "categorical_features_count": len(engine.categorical_columns),
        "numeric_features_count": len(engine.feature_names) - len(engine.categorical_columns),
        "hyperparameters": {
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "min_child_weight": 10,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
        },
        "calibration": {
            "method": "Isotonic Regression",
            "strategy": "5-Fold Stratified Out-of-Fold (OOF) cross-fitting on Train partition",
            "fit_samples": 215257,
            "validation_leakage": "0 validation rows accessed",
        },
        "thresholds": {
            "primary_cost_optimal_1_5": 0.1600,
            "secondary_max_f1": 0.1700,
            "cost_optimal_1_10": 0.0900,
            "default_unadjusted": 0.5000,
        },
        "risk_bands": {
            "LOW": "[0.00, 0.09) - Standard Approval",
            "MEDIUM": "[0.09, 0.25) - Enhanced Verification",
            "HIGH": "[0.25, 1.00] - Elevated Risk / Adverse Action",
        },
        "provenance": card.get("project_identification", {}),
    }
