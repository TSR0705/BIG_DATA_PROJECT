"""Tests for Phase 4 Gold Aggregation & Multi-Table Assembly."""

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

ROOT = pathlib.Path(__file__).resolve().parents[1]
SILVER_BASE_DIR = ROOT / "data" / "silver"
GOLD_BASE_DIR = ROOT / "data" / "gold"
REPORTS_DIR = ROOT / "reports" / "gold"
CONFIGS_DIR = ROOT / "configs"

from src.common.spark_session import get_spark
from src.features.gold import (
    aggregate_bureau,
    aggregate_bureau_balance,
    aggregate_credit_card,
    aggregate_installments,
    aggregate_pos_cash,
    aggregate_previous_application,
    build_app_features,
    join_features_to_spine,
    safe_divide,
    validate_gold_df,
)

EXPECTED_TRAIN_ROWS = 307_511
EXPECTED_TEST_ROWS = 48_744


def gold_data_exists() -> bool:
    """Check if final Gold model input directories exist with Parquet part-files."""
    train_dir = GOLD_BASE_DIR / "model_input"
    test_dir = GOLD_BASE_DIR / "model_input_test"
    if not train_dir.exists() or not list(train_dir.glob("*.parquet")):
        return False
    if not test_dir.exists() or not list(test_dir.glob("*.parquet")):
        return False
    return True


require_gold = pytest.mark.skipif(
    not gold_data_exists(),
    reason="Gold datasets not yet generated; run scripts/run_phase4_gold.py first",
)


@pytest.fixture(scope="session")
def spark():
    """Shared SparkSession for test execution."""
    session = get_spark("TestPhase4Gold")
    yield session


# ======================================================================
# Unit Tests: Synthetic Data Aggregations & Edge Cases
# ======================================================================

def test_synthetic_safe_divide(spark: SparkSession):
    """Verify safe_divide returns NULL for zero, NULL, or NaN denominators without throwing."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("num", DoubleType(), True),
        StructField("den", DoubleType(), True),
    ])
    data = [
        (1, 100.0, 50.0),    # Valid -> 2.0
        (2, 100.0, 0.0),     # Zero denominator -> NULL
        (3, 100.0, None),    # NULL denominator -> NULL
        (4, None, 50.0),     # NULL numerator -> NULL
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


def test_synthetic_bureau_balance_aggregation(spark: SparkSession):
    """Verify bureau_balance aggregates correctly by SK_ID_BUREAU."""
    schema = StructType([
        StructField("SK_ID_BUREAU", IntegerType(), True),
        StructField("MONTHS_BALANCE", IntegerType(), True),
        StructField("STATUS", StringType(), True),
    ])
    data = [
        (1001, -1, "0"),
        (1001, -2, "1"),
        (1001, -3, "C"),
        (1002, -1, "0"),
    ]
    df = spark.createDataFrame(data, schema)
    agg = aggregate_bureau_balance(df)
    rows = {r["SK_ID_BUREAU"]: r for r in agg.collect()}

    assert len(rows) == 2
    assert rows[1001]["BURO_BAL_MONTH_COUNT"] == 3
    assert rows[1001]["BURO_BAL_MONTHS_MIN"] == -3
    assert rows[1001]["BURO_BAL_MONTHS_MAX"] == -1
    assert rows[1001]["BURO_BAL_STATUS_0_COUNT"] == 1
    assert rows[1001]["BURO_BAL_STATUS_1_COUNT"] == 1
    assert rows[1001]["BURO_BAL_STATUS_C_COUNT"] == 1
    assert rows[1001]["BURO_BAL_DPD_COUNT"] == 1
    assert round(rows[1001]["BURO_BAL_DPD_RATIO"], 4) == round(1 / 3, 4)

    assert rows[1002]["BURO_BAL_MONTH_COUNT"] == 1
    assert rows[1002]["BURO_BAL_DPD_COUNT"] == 0
    assert rows[1002]["BURO_BAL_DPD_RATIO"] == 0.0


def test_synthetic_bureau_aggregation(spark: SparkSession):
    """Verify bureau aggregates correctly to SK_ID_CURR grain."""
    bureau_schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("SK_ID_BUREAU", IntegerType(), True),
        StructField("CREDIT_ACTIVE", StringType(), True),
        StructField("DAYS_CREDIT", IntegerType(), True),
        StructField("DAYS_CREDIT_ENDDATE", DoubleType(), True),
        StructField("AMT_CREDIT_SUM", DoubleType(), True),
        StructField("AMT_CREDIT_SUM_DEBT", DoubleType(), True),
        StructField("AMT_CREDIT_SUM_OVERDUE", DoubleType(), True),
        StructField("CNT_CREDIT_PROLONG", IntegerType(), True),
        StructField("AMT_ANNUITY", DoubleType(), True),
    ])
    data = [
        (2001, 1001, "Active", -100, 200.0, 50000.0, 20000.0, 0.0, 0, 1000.0),
        (2001, 1002, "Closed", -500, -100.0, 30000.0, 0.0, 0.0, 1, 500.0),
    ]
    df_bureau = spark.createDataFrame(data, bureau_schema)
    agg = aggregate_bureau(df_bureau)
    rows = {r["SK_ID_CURR"]: r for r in agg.collect()}

    assert len(rows) == 1
    r = rows[2001]
    assert r["BURO_RECORD_COUNT"] == 2
    assert r["BURO_ACTIVE_COUNT"] == 1
    assert r["BURO_CLOSED_COUNT"] == 1
    assert r["BURO_ACTIVE_RATIO"] == 0.5
    assert r["BURO_AMT_CREDIT_SUM_SUM"] == 80000.0
    assert r["BURO_AMT_CREDIT_SUM_DEBT_SUM"] == 20000.0
    assert r["BURO_DEBT_TO_CREDIT_RATIO"] == 0.25
    assert r["BURO_DAYS_CREDIT_MIN"] == -500
    assert r["BURO_DAYS_CREDIT_MAX"] == -100


def test_synthetic_previous_application_aggregation(spark: SparkSession):
    """Verify previous_application aggregates correctly to SK_ID_CURR grain."""
    schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("SK_ID_PREV", IntegerType(), True),
        StructField("NAME_CONTRACT_STATUS", StringType(), True),
        StructField("AMT_CREDIT", DoubleType(), True),
        StructField("AMT_APPLICATION", DoubleType(), True),
        StructField("AMT_ANNUITY", DoubleType(), True),
        StructField("AMT_DOWN_PAYMENT", DoubleType(), True),
        StructField("RATE_DOWN_PAYMENT", DoubleType(), True),
        StructField("DAYS_DECISION", IntegerType(), True),
        StructField("CNT_PAYMENT", DoubleType(), True),
        StructField("FLAG_DAYS_TERMINATION_SENTINEL", IntegerType(), True),
        StructField("FLAG_DAYS_LAST_DUE_SENTINEL", IntegerType(), True),
    ])
    data = [
        (3001, 501, "Approved", 100000.0, 100000.0, 5000.0, 0.0, 0.0, -200, 24.0, 0, 0),
        (3001, 502, "Refused", 50000.0, 80000.0, 4000.0, 0.0, 0.0, -400, 12.0, 1, 0),
    ]
    df_prev = spark.createDataFrame(data, schema)
    agg = aggregate_previous_application(df_prev)
    r = agg.first()

    assert r["SK_ID_CURR"] == 3001
    assert r["PREV_APPLICATION_COUNT"] == 2
    assert r["PREV_APPROVED_COUNT"] == 1
    assert r["PREV_REFUSED_COUNT"] == 1
    assert r["PREV_APPROVAL_RATE"] == 0.5
    assert r["PREV_REFUSAL_RATE"] == 0.5
    assert r["PREV_AMT_CREDIT_SUM"] == 150000.0
    assert r["PREV_AMT_APPLICATION_SUM"] == 180000.0
    assert r["PREV_DAYS_DECISION_MAX"] == -200


def test_synthetic_installments_aggregation(spark: SparkSession):
    """Verify installments_payments calculates delays and aggregates to SK_ID_CURR."""
    schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("SK_ID_PREV", IntegerType(), True),
        StructField("DAYS_INSTALMENT", DoubleType(), True),
        StructField("DAYS_ENTRY_PAYMENT", DoubleType(), True),
        StructField("AMT_INSTALMENT", DoubleType(), True),
        StructField("AMT_PAYMENT", DoubleType(), True),
        StructField("NUM_INSTALMENT_VERSION", DoubleType(), True),
    ])
    data = [
        # Paid 5 days late (delay = 5), full amount paid
        (4001, 601, -30.0, -25.0, 1000.0, 1000.0, 1.0),
        # Paid 2 days early (delay = -2), underpaid by 200
        (4001, 601, -60.0, -62.0, 1000.0, 800.0, 1.0),
    ]
    df_inst = spark.createDataFrame(data, schema)
    agg = aggregate_installments(df_inst)
    r = agg.first()

    assert r["SK_ID_CURR"] == 4001
    assert r["INST_RECORD_COUNT"] == 2
    assert r["INST_AMT_INSTALMENT_SUM"] == 2000.0
    assert r["INST_AMT_PAYMENT_SUM"] == 1800.0
    assert r["INST_AMT_SHORTFALL_SUM"] == 200.0
    assert r["INST_PAYMENT_RATIO"] == 0.9
    assert r["INST_DELAY_MAX"] == 5.0
    assert r["INST_DELAY_MIN"] == -2.0
    assert r["INST_LATE_COUNT"] == 1
    assert r["INST_LATE_RATE"] == 0.5
    assert r["INST_UNDERPAID_COUNT"] == 1
    assert r["INST_UNDERPAID_RATE"] == 0.5


def test_synthetic_join_row_preservation(spark: SparkSession):
    """Verify join_features_to_spine strictly preserves spine row count."""
    spine_schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("AMT_CREDIT", DoubleType(), True),
    ])
    feature_schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("BURO_COUNT", IntegerType(), True),
    ])

    spine_data = [(1, 100.0), (2, 200.0), (3, 300.0)]
    feature_data = [(1, 5), (2, 10)]  # Customer 3 has no bureau history

    spine_df = spark.createDataFrame(spine_data, spine_schema)
    feature_df = spark.createDataFrame(feature_data, feature_schema)

    joined_df, audit = join_features_to_spine(
        spine_df, [("bureau_feat", feature_df)], spine_name="test_spine"
    )

    assert joined_df.count() == 3
    assert audit[0]["row_increase"] == 0
    assert audit[0]["status"] == "PASS"

    rows = {r["SK_ID_CURR"]: r["BURO_COUNT"] for r in joined_df.collect()}
    assert rows[1] == 5
    assert rows[2] == 10
    assert rows[3] is None  # Left join preserved customer 3 with nulls


def test_synthetic_join_row_explosion_detection(spark: SparkSession):
    """Verify join_features_to_spine raises AssertionError if right table contains duplicate keys."""
    spine_schema = StructType([StructField("SK_ID_CURR", IntegerType(), True)])
    feature_schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("VAL", IntegerType(), True),
    ])

    spine_df = spark.createDataFrame([(1,), (2,)], spine_schema)
    # Duplicate key 1 causes Cartesian fan-out on left join
    feature_df = spark.createDataFrame([(1, 10), (1, 20)], feature_schema)

    with pytest.raises(AssertionError, match="Join row explosion detected"):
        join_features_to_spine(spine_df, [("bad_feat", feature_df)], spine_name="test_spine")


# ======================================================================
# Integration Tests: Real Gold Parquet Outputs
# ======================================================================

@require_gold
def test_gold_outputs_exist():
    """Verify model_input and model_input_test exist with Parquet part-files."""
    train_dir = GOLD_BASE_DIR / "model_input"
    test_dir = GOLD_BASE_DIR / "model_input_test"

    assert train_dir.exists(), f"Missing: {train_dir}"
    assert len(list(train_dir.glob("*.parquet"))) > 0

    assert test_dir.exists(), f"Missing: {test_dir}"
    assert len(list(test_dir.glob("*.parquet"))) > 0


@require_gold
def test_gold_train_row_count_and_uniqueness(spark: SparkSession):
    """Verify training Gold dataset has exactly 307,511 rows and 100% unique SK_ID_CURR."""
    df = spark.read.parquet(str(GOLD_BASE_DIR / "model_input"))
    actual_count = df.count()
    assert actual_count == EXPECTED_TRAIN_ROWS, f"Expected {EXPECTED_TRAIN_ROWS:,}, got {actual_count:,}"

    distinct_count = df.select("SK_ID_CURR").distinct().count()
    assert distinct_count == EXPECTED_TRAIN_ROWS, "SK_ID_CURR uniqueness violation in model_input"


@require_gold
def test_gold_test_row_count_and_uniqueness(spark: SparkSession):
    """Verify inference Gold dataset has exactly 48,744 rows and 100% unique SK_ID_CURR."""
    df = spark.read.parquet(str(GOLD_BASE_DIR / "model_input_test"))
    actual_count = df.count()
    assert actual_count == EXPECTED_TEST_ROWS, f"Expected {EXPECTED_TEST_ROWS:,}, got {actual_count:,}"

    distinct_count = df.select("SK_ID_CURR").distinct().count()
    assert distinct_count == EXPECTED_TEST_ROWS, "SK_ID_CURR uniqueness violation in model_input_test"


@require_gold
def test_gold_target_exists_in_train(spark: SparkSession):
    """Verify TARGET column exists in training Gold dataset with valid ground truth values."""
    df = spark.read.parquet(str(GOLD_BASE_DIR / "model_input"))
    assert "TARGET" in df.columns, "TARGET missing in training Gold"

    targets = df.groupBy("TARGET").count().collect()
    target_map = {r["TARGET"]: r["count"] for r in targets}
    assert set(target_map.keys()) == {0, 1}
    assert sum(target_map.values()) == EXPECTED_TRAIN_ROWS


@require_gold
def test_gold_target_not_in_test(spark: SparkSession):
    """Verify TARGET column is strictly absent from test inference dataset."""
    df = spark.read.parquet(str(GOLD_BASE_DIR / "model_input_test"))
    assert "TARGET" not in df.columns, "TARGET illegally present in inference Gold dataset"


@require_gold
def test_no_duplicate_columns(spark: SparkSession):
    """Verify neither train nor test Gold datasets contain duplicate column names."""
    for sub in ["model_input", "model_input_test"]:
        df = spark.read.parquet(str(GOLD_BASE_DIR / sub))
        cols = df.columns
        assert len(cols) == len(set(cols)), f"Duplicate columns in {sub}: {len(cols)} vs {len(set(cols))}"


@require_gold
def test_gold_feature_count_range(spark: SparkSession):
    """Verify total Gold features generated falls within the recommended 250–350 range."""
    df = spark.read.parquet(str(GOLD_BASE_DIR / "model_input"))
    total_features = len(df.columns)
    # 250 to 350 range
    assert 220 <= total_features <= 350, f"Expected ~250-350 features, found {total_features}"


@require_gold
def test_feature_registry_exists():
    """Verify configs/features.yaml exists and documents all Gold features."""
    yaml_path = CONFIGS_DIR / "features.yaml"
    assert yaml_path.exists(), f"Missing feature registry: {yaml_path}"

    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    assert isinstance(registry, dict)
    assert len(registry) >= 200
    assert "SK_ID_CURR" in registry
    assert "TARGET" in registry
    assert "APP_PAYMENT_RATE" in registry
    assert "BURO_RECORD_COUNT" in registry
    assert "PREV_APPLICATION_COUNT" in registry
    assert "INST_LATE_RATE" in registry


@require_gold
def test_join_audit_passes():
    """Verify reports/gold/join_audit.json contains passing audits with zero row increases."""
    audit_path = REPORTS_DIR / "join_audit.json"
    assert audit_path.exists(), f"Missing join audit: {audit_path}"

    import json
    with open(audit_path, encoding="utf-8") as f:
        audits = json.load(f)

    assert len(audits) >= 10
    for a in audits:
        assert a["status"] == "PASS"
        assert a["row_increase"] == 0


@require_gold
def test_gold_summary_markdown_exists():
    """Verify reports/gold/summary.md and physical_plan.txt exist."""
    assert (REPORTS_DIR / "summary.md").exists()
    assert (REPORTS_DIR / "physical_plan.txt").exists()
