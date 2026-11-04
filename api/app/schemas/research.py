"""Research metrics and subgroup evaluation schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ModelBenchmarkRow(BaseModel):
    model_name: str
    configuration: str
    threshold: float
    calibration: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    ece: float
    recall: float
    precision: float
    f1_score: float


class SubgroupRow(BaseModel):
    dimension: str
    subgroup_name: str
    sample_size: int
    defaults: int
    default_rate: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    brier_score: float
    ece: float
    precision: float
    recall: float


class ResearchMetricsResponse(BaseModel):
    project: str
    validation_performance: Dict[str, Any]
    test_evaluation: Dict[str, Any]
    benchmarks: List[ModelBenchmarkRow]
    subgroups: List[SubgroupRow]
