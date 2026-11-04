"""CreditLens Inference Engine.

Provides leak-free, production-grade inference using:
- Frozen Phase 8 Tuned XGBoost base model
- Phase 11 Independent Out-of-Fold Isotonic Calibrator
- Fitted Tree Preprocessor (OrdinalEncoder for 16 categorical features)
- Verified feature schema (256 features: 16 categorical, 240 numeric)
- Frozen operating threshold (tau = 0.1600) and regulatory risk bands
"""

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "artifacts" / "models"
PREPROCESSING_DIR = ROOT / "artifacts" / "preprocessing"


class CreditLensInferenceEngine:
    """Production inference engine for CreditLens loan default risk scoring."""

    def __init__(
        self,
        model_path: Optional[pathlib.Path] = None,
        calibrator_path: Optional[pathlib.Path] = None,
        preprocessor_path: Optional[pathlib.Path] = None,
    ):
        self.model_path = model_path or (MODELS_DIR / "phase8_best_xgboost.joblib")
        self.calibrator_path = calibrator_path or (MODELS_DIR / "phase11_independent_calibrator.joblib")
        self.preprocessor_path = preprocessor_path or (MODELS_DIR / "tree_preprocessor.joblib")

        self._base_model = None
        self._calibrator = None
        self._preprocessor = None
        self._feature_names = None
        self._cat_columns = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all frozen model and preprocessing artifacts into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Base model not found at {self.model_path}")
        if not self.calibrator_path.exists():
            raise FileNotFoundError(f"Calibrator not found at {self.calibrator_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")

        self._base_model = joblib.load(self.model_path)
        self._calibrator = joblib.load(self.calibrator_path)
        self._preprocessor = joblib.load(self.preprocessor_path)

        self._feature_names = list(self._base_model.feature_names_in_)
        self._cat_columns = list(self._preprocessor.feature_names_in_)

    @property
    def feature_names(self) -> List[str]:
        """Return the 256 expected feature names in exact model order."""
        return self._feature_names.copy()

    @property
    def categorical_columns(self) -> List[str]:
        """Return the 16 categorical feature names."""
        return self._cat_columns.copy()

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw input DataFrame into preprocessed 256-feature model input."""
        # Ensure all required features are present
        missing = [col for col in self._feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing {len(missing)} required features: {missing[:10]}...")

        # Create copy and order columns
        X = df[self._feature_names].copy()

        # Apply ordinal encoding to the 16 categorical features
        X[self._cat_columns] = self._preprocessor.transform(X[self._cat_columns])

        return X

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Compute calibrated loan default probabilities for the input DataFrame.
        
        Returns:
            np.ndarray: Array of calibrated probabilities for class 1 (default), shape (n_samples,).
        """
        X = self.preprocess_features(df)
        probs = self._calibrator.predict_proba(X)
        return probs[:, 1]

    def predict_risk_tier(self, prob: float) -> str:
        """Assign risk tier based on frozen project definitions:
        - LOW: [0.00, 0.09)
        - MEDIUM: [0.09, 0.25)
        - HIGH: [0.25, 1.00]
        """
        if prob < 0.09:
            return "LOW"
        elif prob < 0.25:
            return "MEDIUM"
        else:
            return "HIGH"

    def predict_decision(self, prob: float, threshold: float = 0.1600) -> Tuple[int, str]:
        """Classify loan application under frozen cost-optimal threshold (tau = 0.1600).
        
        Returns:
            Tuple[int, str]: (binary_prediction, decision_label: 'APPROVE' or 'REJECT')
        """
        pred = 1 if prob >= threshold else 0
        label = "REJECT" if pred == 1 else "APPROVE"
        return pred, label

    def score_applicant(
        self,
        applicant_data: Union[pd.Series, Dict[str, Any]],
        applicant_id: Optional[str] = None,
        threshold: float = 0.1600,
    ) -> Dict[str, Any]:
        """Score a single loan applicant and return detailed risk analytics."""
        if isinstance(applicant_data, dict):
            df = pd.DataFrame([applicant_data])
        elif isinstance(applicant_data, pd.Series):
            df = pd.DataFrame([applicant_data.to_dict()])
        else:
            raise TypeError("applicant_data must be a dict or pandas Series")

        prob = float(self.predict_proba(df)[0])
        tier = self.predict_risk_tier(prob)
        pred, decision = self.predict_decision(prob, threshold=threshold)

        return {
            "applicant_id": applicant_id or "APP-PREDICT",
            "default_probability": round(prob, 4),
            "default_percentage": round(prob * 100, 2),
            "decision": decision,
            "binary_prediction": pred,
            "risk_tier": tier,
            "threshold_used": threshold,
            "model_version": "XGBoost Tuned (Phase 8) + Isotonic OOF Calibrator (Phase 11)",
        }
