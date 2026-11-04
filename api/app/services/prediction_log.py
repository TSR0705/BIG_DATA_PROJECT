"""Append-only log of real model predictions made through the API.

Every score produced by the frozen inference engine (Check Loan Risk page,
applicant search, sample scoring) is appended here, so the dashboard's
"Recent Predictions" card shows what the model actually predicted and when.
Nothing in this file is invented: when the log is empty it is seeded by
scoring real applicants from the frozen test split with the real engine.
"""

import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

from api.app.core.config import PREDICTION_LOG_PATH, PREDICTIONS_DIR

_LOCK = threading.Lock()
RISK_TIER_LABELS = {"LOW": "Low Risk", "MEDIUM": "Medium Risk", "HIGH": "High Risk"}


def record_prediction(score: Dict[str, Any], source: str, sk_id_curr: int | None = None) -> Dict[str, Any]:
    """Append one engine result to the log and return the stored entry."""
    entry = {
        "applicant_id": score["applicant_id"],
        "sk_id_curr": sk_id_curr,
        "probability": float(score["default_probability"]),
        "risk_tier": RISK_TIER_LABELS.get(score["risk_tier"], score["risk_tier"]),
        "decision": score["decision"],
        "threshold_used": score["threshold_used"],
        "model_version": score["model_version"],
        "source": source,  # "check_risk" | "search" | "sample"
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with open(PREDICTION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    return entry


def read_predictions(limit: int = 5) -> List[Dict[str, Any]]:
    """Return the most recent `limit` logged predictions, newest first."""
    if not PREDICTION_LOG_PATH.exists():
        return []
    with _LOCK:
        with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    entries = [json.loads(ln) for ln in lines[-limit:]]
    return list(reversed(entries))


def count_predictions() -> int:
    if not PREDICTION_LOG_PATH.exists():
        return 0
    with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())
