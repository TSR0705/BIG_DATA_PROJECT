"""Tests for Phase 5 Feature Engineering & Leakage Audit Layer."""

import math
import pathlib
import pytest
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_INPUT_DIR = ROOT / "data" / "model_input"
MODEL_INPUT_TEST_DIR = ROOT / "data" / "model_input_test"
MODEL_INPUT_PATH = (MODEL_INPUT_DIR / "model_input.parquet") if (MODEL_INPUT_DIR / "model_input.parquet").exists() else MODEL_INPUT_DIR
MODEL_INPUT_TEST_PATH = (MODEL_INPUT_TEST_DIR / "model_input_test.parquet") if (MODEL_INPUT_TEST_DIR / "model_input_test.parquet").exists() else MODEL_INPUT_TEST_DIR
CONFIGS_DIR = ROOT / "configs"
REPORTS_DIR = ROOT / "reports"

from src.common.spark_session import get_spark
from src.features.feature_engineering import (
    audit_feature_quality,
    audit_leakage,
    build_application_derived_features,
    build_historical_derived_features,
    safe_divide,
    sanitize_numeric_column,
    update_feature_registry,
    validate_feature_schema,
)

EXPECTED_TRAIN_ROWS = 307_511
EXPECTED_TEST_ROWS = 48_744


def model_input_exists() -> bool:
    """Check if Phase 5 final model input datasets exist with Parquet part-files."""
    if not MODEL_INPUT_DIR.exists() or not list(MODEL_INPUT_DIR.rglob("*.parquet")):
        return False
    if not MODEL_INPUT_TEST_DIR.exists() or not list(MODEL_INPUT_TEST_DIR.rglob("*.parquet")):
        return False
    return True


require_model_input = pytest.mark.skipif(
    not model_input_exists(),
    reason="Final model input Parquet datasets not yet generated; run scripts/run_phase5_features.py first",
)


@pytest.fixture(scope="session")
def spark():
    """Shared SparkSession for test execution."""
    session = get_spark("TestPhase5Features")
    yield session


# ======================================================================
# Unit Tests: Synthetic Data Operations & Edge Cases
# ======================================================================

def test_synthetic_safe_ratio(spark: SparkSession):
    """Verify safe_divide returns NULL for zero, NULL, or NaN denominators without throwing."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("num", DoubleType(), True),
        StructField("den", DoubleType(), True),
    ])
    data = [
        (1, 100.0, 50.0),          # Normal -> 2.0
        (2, 100.0, 0.0),           # Zero denominator -> NULL
        (3, 100.0, None),          # NULL denominator -> NULL
        (4, None, 50.0),           # NULL numerator -> NULL
        (5, 100.0, float("nan")),  # NaN denominator -> NULL
    ]
    df = spark.createDataFrame(data, schema)
    res_df = df.withColumn("ratio", safe_divide(F.col("num"), F.col("den")))
    rows = {r["id"]: r["ratio"] for r in res_df.collect()}

    assert rows[1] == 2.0
    assert rows[2] is None
    assert rows[3] is None
    assert rows[4] is None
    assert rows[5] is None


def test_synthetic_age_conversion(spark: SparkSession):
    """Verify age conversion from negative days offset produces valid approximate years."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("DAYS_BIRTH", IntegerType(), True),
    ])
    # -14610 days ≈ 40.0 years
    df = spark.createDataFrame([(1, -14610), (2, -7305)], schema)
    res = df.withColumn("AGE_YEARS", F.abs(F.col("DAYS_BIRTH")) / 365.25).collect()
    rows = {r["id"]: r["AGE_YEARS"] for r in res}

    assert round(rows[1], 1) == 40.0
    assert round(rows[2], 1) == 20.0


def test_synthetic_missingness_count(spark: SparkSession):
    """Verify missingness features sum nulls across designated baseline columns excluding ID/TARGET."""
    schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("TARGET", IntegerType(), True),
        StructField("A", DoubleType(), True),
        StructField("B", DoubleType(), True),
        StructField("C", StringType(), True),
    ])
    data = [
        (101, 1, 10.0, None, None),   # A present, B null, C null -> 2 missing
        (102, 0, 10.0, 20.0, "X"),    # 0 missing
    ]
    df = spark.createDataFrame(data, schema)
    raw_cols = ["A", "B", "C"]  # strictly excluding SK_ID_CURR and TARGET
    df_res = build_application_derived_features(df, raw_cols)
    rows = {r["SK_ID_CURR"]: r for r in df_res.collect()}

    assert rows[101]["APP_MISSING_COUNT"] == 2
    assert rows[101]["APP_MISSING_RATIO"] == 2 / 3.0
    assert rows[102]["APP_MISSING_COUNT"] == 0
    assert rows[102]["APP_MISSING_RATIO"] == 0.0


def test_synthetic_historical_derived_features(spark: SparkSession):
    """Verify derived historical ratios and discipline score calculations."""
    schema = StructType([
        StructField("AMT_CREDIT", DoubleType(), True),
        StructField("AMT_INCOME_TOTAL", DoubleType(), True),
        StructField("BURO_AMT_CREDIT_SUM_DEBT_SUM", DoubleType(), True),
        StructField("BURO_AMT_CREDIT_SUM_SUM", DoubleType(), True),
        StructField("BURO_AMT_CREDIT_SUM_OVERDUE_SUM", DoubleType(), True),
        StructField("PREV_AMT_CREDIT_SUM", DoubleType(), True),
        StructField("INST_LATE_COUNT", IntegerType(), True),
        StructField("INST_PAYMENT_RATIO", DoubleType(), True),
        StructField("INST_LATE_RATE", DoubleType(), True),
        StructField("INST_AMT_PAYMENT_SUM", DoubleType(), True),
        StructField("CC_AMT_BALANCE_MEAN", DoubleType(), True),
        StructField("CC_DPD_MONTH_COUNT", IntegerType(), True),
        StructField("POS_DPD_MONTH_COUNT", IntegerType(), True),
    ])
    data = [
        (
            100000.0,  # AMT_CREDIT
            50000.0,   # AMT_INCOME_TOTAL
            20000.0,   # BURO DEBT
            80000.0,   # BURO CREDIT SUM
            0.0,       # BURO OVERDUE
            50000.0,   # PREV CREDIT
            2,         # INST LATE COUNT
            0.95,      # INST PAYMENT RATIO
            0.10,      # INST LATE RATE (10%)
            45000.0,   # INST PAYMENT SUM
            5000.0,    # CC BALANCE MEAN
            0,         # CC DPD COUNT
            0,         # POS DPD COUNT
        )
    ]
    df = spark.createDataFrame(data, schema)
    res = build_historical_derived_features(df).first()

    # Total Debt / Income: (20,000 + 100,000) / 50,000 = 2.4
    assert res["DERIVED_TOTAL_DEBT_TO_INCOME_RATIO"] == 2.4

    # Buro to Current Credit: 80,000 / 100,000 = 0.8
    assert res["DERIVED_BURO_TO_CURRENT_CREDIT_RATIO"] == 0.8

    # Overall overdue flag: INST_LATE_COUNT is 2 > 0 -> flag = 1
    assert res["DERIVED_OVERALL_OVERDUE_FLAG"] == 1

    # Payment Discipline: 0.95 * (1 - 0.10) = 0.95 * 0.90 = 0.855
    assert round(res["DERIVED_PAYMENT_DISCIPLINE_SCORE"], 3) == 0.855


def test_synthetic_schema_consistency_validator(spark: SparkSession):
    """Verify validate_feature_schema detects exact schema parity and mismatches."""
    schema_train = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("TARGET", IntegerType(), True),
        StructField("FEAT1", DoubleType(), True),
    ])
    schema_test_valid = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("FEAT1", DoubleType(), True),
    ])
    schema_test_invalid = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("FEAT1", StringType(), True),  # Type mismatch
    ])

    df_train = spark.createDataFrame([], schema_train)
    df_test_valid = spark.createDataFrame([], schema_test_valid)
    df_test_invalid = spark.createDataFrame([], schema_test_invalid)

    audit_valid = validate_feature_schema(df_train, df_test_valid)
    assert audit_valid["schema_match"] is True
    assert audit_valid["type_match"] is True

    audit_invalid = validate_feature_schema(df_train, df_test_invalid)
    assert audit_invalid["type_match"] is False


def test_synthetic_leakage_audit_detection(spark: SparkSession):
    """Verify audit_leakage flags illegal target-derived column names."""
    schema_clean = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("TARGET", IntegerType(), True),
        StructField("APP_RATIO", DoubleType(), True),
    ])
    schema_leaky = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("TARGET", IntegerType(), True),
        StructField("TARGET_ENCODED_OCCUPATION", DoubleType(), True),  # Illegal!
    ])
    test_schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("APP_RATIO", DoubleType(), True),
    ])

    df_clean = spark.createDataFrame([], schema_clean)
    df_leaky = spark.createDataFrame([], schema_leaky)
    df_test = spark.createDataFrame([], test_schema)

    clean_audit = audit_leakage(df_clean, df_test)
    assert clean_audit["overall_leakage_status"] == "PASS"

    leaky_audit = audit_leakage(df_leaky, df_test)
    assert leaky_audit["overall_leakage_status"] == "FAIL"
    assert "TARGET_ENCODED_OCCUPATION" in leaky_audit["target_derived_columns"]


def test_synthetic_invalid_numeric_sanitizer(spark: SparkSession):
    """Verify sanitize_numeric_column removes NaN and Infinity values."""
    schema = StructType([
        StructField("val", DoubleType(), True),
    ])
    data = [(10.0,), (float("nan"),), (float("inf"),), (float("-inf"),), (None,)]
    df = spark.createDataFrame(data, schema)
    res_df = df.withColumn("clean_val", sanitize_numeric_column(F.col("val")))
    rows = [r["clean_val"] for r in res_df.collect()]

    assert rows[0] == 10.0
    assert rows[1] is None
    assert rows[2] is None
    assert rows[3] is None
    assert rows[4] is None


# ======================================================================
# Integration Tests: Real Final Model Input Datasets
# ======================================================================

@require_model_input
def test_model_input_outputs_exist():
    """Verify final model input Parquet files exist."""
    assert MODEL_INPUT_DIR.exists()
    assert len(list(MODEL_INPUT_DIR.rglob("*.parquet"))) > 0

    assert MODEL_INPUT_TEST_DIR.exists()
    assert len(list(MODEL_INPUT_TEST_DIR.rglob("*.parquet"))) > 0


@require_model_input
def test_model_input_train_row_count_and_uniqueness(spark: SparkSession):
    """Verify training dataset has exactly 307,511 rows and 100% unique SK_ID_CURR."""
    df = spark.read.parquet(str(MODEL_INPUT_PATH))
    cnt = df.count()
    assert cnt == EXPECTED_TRAIN_ROWS, f"Expected {EXPECTED_TRAIN_ROWS:,}, got {cnt:,}"

    distinct_cnt = df.select("SK_ID_CURR").distinct().count()
    assert distinct_cnt == EXPECTED_TRAIN_ROWS, "SK_ID_CURR uniqueness violation in model_input"


@require_model_input
def test_model_input_test_row_count_and_uniqueness(spark: SparkSession):
    """Verify inference dataset has exactly 48,744 rows and 100% unique SK_ID_CURR."""
    df = spark.read.parquet(str(MODEL_INPUT_TEST_PATH))
    cnt = df.count()
    assert cnt == EXPECTED_TEST_ROWS, f"Expected {EXPECTED_TEST_ROWS:,}, got {cnt:,}"

    distinct_cnt = df.select("SK_ID_CURR").distinct().count()
    assert distinct_cnt == EXPECTED_TEST_ROWS, "SK_ID_CURR uniqueness violation in model_input_test"


@require_model_input
def test_target_exists_only_in_train(spark: SparkSession):
    """Verify TARGET column exists in train and is strictly absent from test."""
    train_df = spark.read.parquet(str(MODEL_INPUT_PATH))
    test_df = spark.read.parquet(str(MODEL_INPUT_TEST_PATH))

    assert "TARGET" in train_df.columns, "TARGET missing in model_input"
    assert "TARGET" not in test_df.columns, "TARGET illegally present in model_input_test"


@require_model_input
def test_train_test_feature_parity(spark: SparkSession):
    """Verify train and test share exact identical feature sets (except TARGET in train)."""
    train_df = spark.read.parquet(str(MODEL_INPUT_PATH))
    test_df = spark.read.parquet(str(MODEL_INPUT_TEST_PATH))

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    diff = train_cols - test_cols
    assert diff == {"TARGET"}, f"Unexpected difference between train and test: {diff}"
    assert len(test_cols - train_cols) == 0


@require_model_input
def test_no_invalid_numerical_values(spark: SparkSession):
    """Verify zero NaN, Inf, or -Inf values across final model input datasets."""
    train_df = spark.read.parquet(str(MODEL_INPUT_PATH))
    test_df = spark.read.parquet(str(MODEL_INPUT_TEST_PATH))

    train_dq = audit_feature_quality(train_df, "train")
    test_dq = audit_feature_quality(test_df, "test")

    assert train_dq["has_invalid_values"] is False, f"Invalid numbers in train: {train_dq['invalid_numeric_columns']}"
    assert test_dq["has_invalid_values"] is False, f"Invalid numbers in test: {test_dq['invalid_numeric_columns']}"


@require_model_input
def test_feature_registry_complete():
    """Verify configs/features.yaml documents all features with required schema fields."""
    yaml_path = CONFIGS_DIR / "features.yaml"
    assert yaml_path.exists(), f"Missing feature registry: {yaml_path}"

    with open(yaml_path, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    assert isinstance(registry, dict)
    assert len(registry) >= 240

    # Check required schema fields
    for feat_name, meta in registry.items():
        assert "family" in meta, f"Missing 'family' in {feat_name}"
        assert "source" in meta, f"Missing 'source' in {feat_name}"
        assert "definition" in meta, f"Missing 'definition' in {feat_name}"
        assert "description" in meta, f"Missing 'description' in {feat_name}"
        assert "dtype" in meta, f"Missing 'dtype' in {feat_name}"
        assert "null_behavior" in meta, f"Missing 'null_behavior' in {feat_name}"
        assert "leakage_status" in meta, f"Missing 'leakage_status' in {feat_name}"
        assert "excluded_from_model" in meta, f"Missing 'excluded_from_model' in {feat_name}"

    # Verify ID and LABEL exclusion flags
    assert registry["SK_ID_CURR"]["excluded_from_model"] is True
    assert registry["SK_ID_CURR"]["family"] == "ID"
    assert registry["TARGET"]["excluded_from_model"] is True
    assert registry["TARGET"]["family"] == "LABEL"


@require_model_input
def test_leakage_audit_passes():
    """Verify reports/leakage_audit.json reports overall PASS with zero violations."""
    json_path = REPORTS_DIR / "leakage_audit.json"
    assert json_path.exists(), f"Missing leakage audit report: {json_path}"

    import json
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("overall_leakage_status") == "PASS"
    assert data.get("target_rule_pass") is True
    assert data.get("id_rule_pass") is True
    assert data.get("target_derived_pass") is True
    assert len(data.get("target_derived_columns", [])) == 0


@require_model_input
def test_feature_quality_reports_exist():
    """Verify reports/feature_quality.* exist."""
    assert (REPORTS_DIR / "feature_quality.json").exists()
    assert (REPORTS_DIR / "feature_quality.md").exists()
    assert (REPORTS_DIR / "leakage_audit.md").exists()
