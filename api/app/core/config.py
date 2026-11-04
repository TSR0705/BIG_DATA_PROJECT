"""Core configuration settings for CreditLens API."""

import pathlib
from typing import List

ROOT = pathlib.Path(__file__).resolve().parents[3]

PROJECT_NAME: str = "CreditLens API"
VERSION: str = "1.1.0"
DESCRIPTION: str = "CreditLens - Loan Default Prediction API using Big Data Analytics & Explainable AI"

# CORS Origins
ALLOWED_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# File Paths
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EXPLAINABILITY_DIR = ARTIFACTS_DIR / "explainability"
REPORTS_DIR = ROOT / "reports"
SPLITS_DIR = DATA_DIR / "splits"
MODEL_INPUT_PATH = DATA_DIR / "model_input" / "model_input.parquet"
MODEL_INPUT_TEST_PATH = DATA_DIR / "model_input_test" / "model_input_test.parquet"
MODEL_CARD_PATH = ROOT / "models" / "model_card.json"

# Verified research artifacts consumed by the dashboard
SPLIT_REPORT_PATH = REPORTS_DIR / "split_report.json"
SUBGROUP_ANALYSIS_PATH = REPORTS_DIR / "subgroup_analysis.csv"          # Phase 11 validation metrics (calibrated model)
SHAP_GLOBAL_IMPORTANCE_PATH = REPORTS_DIR / "shap_global_importance.csv"  # Phase 9 global SHAP (validation sample)
SHAP_TEST_VALUES_PATH = EXPLAINABILITY_DIR / "test_split_shap_values.parquet"  # Phase 11 SHAP matrix, frozen test split
SHAP_TEST_METADATA_PATH = EXPLAINABILITY_DIR / "test_split_shap_metadata.json"
PHASE11_CALIBRATION_METADATA_PATH = MODELS_DIR / "phase11_calibration_metadata.json"

# Runtime (non-versioned) data written by the API
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
PREDICTION_LOG_PATH = PREDICTIONS_DIR / "prediction_log.jsonl"
DASHBOARD_CACHE_DIR = ARTIFACTS_DIR / "dashboard_cache"

# Frozen decision policy (Phase 8 cost-optimal 1:5 threshold, Phase 9/11 risk bands)
OPERATING_THRESHOLD: float = 0.16
RISK_BANDS = {"LOW": (0.0, 0.09), "MEDIUM": (0.09, 0.25), "HIGH": (0.25, 1.0)}
