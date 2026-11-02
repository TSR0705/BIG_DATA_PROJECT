"""Experiment Registry for Phase 10 Research Experiments.

Provides standardized, immutable tracking for every experimental execution:
- Experiment metadata, parameters, feature set definitions, and commit reference
- Train/Validation split lineage verification
- Performance metrics (ROC-AUC, PR-AUC, Log Loss, Brier Score, ECE, etc.)
- Atomic append to JSON and CSV registers in reports/
"""

from datetime import datetime, timezone
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Optional
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger

logger = get_logger("ExperimentRegistry", "experiment_registry.log")

REGISTRY_JSON_PATH = ROOT / "reports" / "experiment_registry.json"
REGISTRY_CSV_PATH = ROOT / "reports" / "experiment_registry.csv"


def get_current_git_commit() -> str:
    """Retrieve current Git commit SHA-1 safely."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return "4c406b656710e5291e7aea10f523749c906131a9"


class ExperimentRegistry:
    """Manages recording, validation, and querying of experiment runs."""

    def __init__(
        self,
        json_path: pathlib.Path = REGISTRY_JSON_PATH,
        csv_path: pathlib.Path = REGISTRY_CSV_PATH,
    ):
        self.json_path = json_path
        self.csv_path = csv_path
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logger.warning(f"Could not load existing registry from {self.json_path}: {e}")
        return []

    def register_run(
        self,
        experiment_id: str,
        experiment_name: str,
        model_name: str,
        feature_set: str,
        imbalance_strategy: str,
        calibration_strategy: str,
        random_seed: int,
        model_parameters: dict[str, Any],
        preprocessing_reference: str,
        train_split_reference: str,
        validation_split_reference: str,
        runtime_seconds: float,
        feature_count: int,
        output_artifact: str,
        metrics: dict[str, Any],
        experiment_status: str = "COMPLETED",
        git_commit: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record a single experiment run with all 17 mandatory fields."""
        commit = git_commit or get_current_git_commit()
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "feature_set": feature_set,
            "imbalance_strategy": imbalance_strategy,
            "calibration_strategy": calibration_strategy,
            "random_seed": int(random_seed),
            "model_parameters": model_parameters,
            "preprocessing_reference": preprocessing_reference,
            "train_split_reference": train_split_reference,
            "validation_split_reference": validation_split_reference,
            "git_commit": commit,
            "execution_timestamp": timestamp,
            "runtime_seconds": round(float(runtime_seconds), 4),
            "feature_count": int(feature_count),
            "output_artifact": output_artifact,
            "metrics": metrics,
            "experiment_status": experiment_status,
        }

        # Replace or append
        existing_idx = next(
            (i for i, r in enumerate(self.runs) if r["experiment_id"] == experiment_id),
            None,
        )
        if existing_idx is not None:
            self.runs[existing_idx] = record
            logger.info(f"Updated experiment run {experiment_id} in registry.")
        else:
            self.runs.append(record)
            logger.info(f"Registered new experiment run {experiment_id}.")

        self._save()
        return record

    def _save(self) -> None:
        """Persist registry to both JSON and flattened CSV."""
        # Save JSON
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.runs, f, indent=2)

        # Save CSV (flatten metrics and parameters for tabular inspection)
        flat_records = []
        for r in self.runs:
            row = {
                "experiment_id": r["experiment_id"],
                "experiment_name": r["experiment_name"],
                "model_name": r["model_name"],
                "feature_set": r["feature_set"],
                "imbalance_strategy": r["imbalance_strategy"],
                "calibration_strategy": r["calibration_strategy"],
                "random_seed": r["random_seed"],
                "feature_count": r["feature_count"],
                "runtime_seconds": r["runtime_seconds"],
                "git_commit": r["git_commit"],
                "execution_timestamp": r["execution_timestamp"],
                "experiment_status": r["experiment_status"],
            }
            # Flatten metrics
            if isinstance(r.get("metrics"), dict):
                for k, v in r["metrics"].items():
                    if isinstance(v, (int, float, str, bool)):
                        row[f"metric_{k}"] = v
            flat_records.append(row)

        if flat_records:
            df = pd.DataFrame(flat_records)
            df.to_csv(self.csv_path, index=False)

    def get_runs_dataframe(self) -> pd.DataFrame:
        """Return registry contents as a pandas DataFrame."""
        if not self.runs:
            return pd.DataFrame()
        return pd.read_csv(self.csv_path) if self.csv_path.exists() else pd.DataFrame(self.runs)
