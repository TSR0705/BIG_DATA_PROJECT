"""Tests for Phase 3 Silver Cleaning & Data Quality Layer."""

import json
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
BRONZE_BASE_DIR = ROOT / "data" / "bronze"
SILVER_BASE_DIR = ROOT / "data" / "silver"
REPORTS_DIR = ROOT / "reports" / "silver"
DATASET_DIR = ROOT / "dataset"

from configs.tables import TABLES
from src.common.spark_session import get_spark
from src.preprocessing.silver import (
    APPLICATION_SENTINELS,
    APPLICATION_XNA_COLUMNS,
    POS_CASH_XNA_COLUMNS,
    PREVIOUS_APPLICATION_SENTINELS,
    PREVIOUS_APPLICATION_XNA_COLUMNS,
    SENTINEL_VALUE,
    clean_application,
    clean_categorical_xna,
    clean_pos_cash_balance,
    clean_previous_application,
    handle_sentinel,
)

EXPECTED_TABLES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "credit_card_balance",
    "POS_CASH_balance",
]

EXPECTED_ROW_COUNTS = {
    "application_train": 307_511,
    "application_test": 48_744,
    "bureau": 1_716_428,
    "bureau_balance": 27_299_925,
    "previous_application": 1_670_214,
    "installments_payments": 13_605_401,
    "credit_card_balance": 3_840_312,
    "POS_CASH_balance": 10_001_358,
}


def silver_data_exists() -> bool:
    """Check if all 8 Silver Parquet table directories exist with part files."""
    for tbl in EXPECTED_TABLES:
        s_dir = SILVER_BASE_DIR / tbl
        if not s_dir.exists() or not list(s_dir.glob("*.parquet")):
            return False
    return True


require_silver = pytest.mark.skipif(
    not silver_data_exists(),
    reason="Silver Parquet tables not yet generated; run scripts/run_phase3_silver.py first",
)


@pytest.fixture(scope="session")
def spark():
    """Shared SparkSession for test execution."""
    session = get_spark("TestPhase3Silver")
    yield session


# ======================================================================
# Unit Tests: Synthetic Data Transformations
# ======================================================================

def test_synthetic_sentinel_replacement(spark: SparkSession):
    """Verify handle_sentinel replaces 365243 with NULL and sets flag=1 while preserving negatives and NULLs."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("DAYS_EMPLOYED", IntegerType(), True),
    ])
    data = [
        (1, 365243),   # Sentinel
        (2, -1000),    # Normal valid negative offset
        (3, None),     # Already null
        (4, 0),        # Zero offset
    ]
    df = spark.createDataFrame(data, schema)
    cleaned_df = handle_sentinel(df, "DAYS_EMPLOYED", sentinel_val=365243, flag_col="FLAG_DAYS_EMPLOYED_SENTINEL")

    rows = {r["id"]: r for r in cleaned_df.collect()}

    # Sentinel row: 365243 -> NULL, flag -> 1
    assert rows[1]["DAYS_EMPLOYED"] is None
    assert rows[1]["FLAG_DAYS_EMPLOYED_SENTINEL"] == 1

    # Valid negative offset: preserved unchanged, flag -> 0
    assert rows[2]["DAYS_EMPLOYED"] == -1000
    assert rows[2]["FLAG_DAYS_EMPLOYED_SENTINEL"] == 0

    # Original null: preserved null, flag -> 0
    assert rows[3]["DAYS_EMPLOYED"] is None
    assert rows[3]["FLAG_DAYS_EMPLOYED_SENTINEL"] == 0

    # Zero: preserved zero, flag -> 0
    assert rows[4]["DAYS_EMPLOYED"] == 0
    assert rows[4]["FLAG_DAYS_EMPLOYED_SENTINEL"] == 0


def test_synthetic_xna_replacement(spark: SparkSession):
    """Verify clean_categorical_xna replaces 'XNA' with NULL and sets flag=1."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("CODE_GENDER", StringType(), True),
    ])
    data = [
        (1, "XNA"),
        (2, "M"),
        (3, "F"),
        (4, None),
    ]
    df = spark.createDataFrame(data, schema)
    cleaned_df = clean_categorical_xna(df, "CODE_GENDER", flag_col="FLAG_CODE_GENDER_XNA")

    rows = {r["id"]: r for r in cleaned_df.collect()}

    # XNA row: 'XNA' -> NULL, flag -> 1
    assert rows[1]["CODE_GENDER"] is None
    assert rows[1]["FLAG_CODE_GENDER_XNA"] == 1

    # Normal values: unchanged, flag -> 0
    assert rows[2]["CODE_GENDER"] == "M"
    assert rows[2]["FLAG_CODE_GENDER_XNA"] == 0

    assert rows[3]["CODE_GENDER"] == "F"
    assert rows[3]["FLAG_CODE_GENDER_XNA"] == 0

    # NULL: unchanged, flag -> 0
    assert rows[4]["CODE_GENDER"] is None
    assert rows[4]["FLAG_CODE_GENDER_XNA"] == 0


def test_synthetic_clean_application(spark: SparkSession):
    """Verify clean_application handles DAYS_EMPLOYED and XNA fields simultaneously."""
    schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), True),
        StructField("TARGET", IntegerType(), True),
        StructField("DAYS_EMPLOYED", IntegerType(), True),
        StructField("CODE_GENDER", StringType(), True),
        StructField("ORGANIZATION_TYPE", StringType(), True),
    ])
    data = [
        (101, 1, 365243, "XNA", "XNA"),
        (102, 0, -2500, "F", "Business Entity Type 3"),
    ]
    df = spark.createDataFrame(data, schema)
    cleaned = clean_application(df, table_name="application_train")

    assert "FLAG_DAYS_EMPLOYED_SENTINEL" in cleaned.columns
    assert "FLAG_CODE_GENDER_XNA" in cleaned.columns
    assert "FLAG_ORGANIZATION_TYPE_XNA" in cleaned.columns

    rows = {r["SK_ID_CURR"]: r for r in cleaned.collect()}

    # Row 101 had all 3 anomalies
    assert rows[101]["DAYS_EMPLOYED"] is None
    assert rows[101]["FLAG_DAYS_EMPLOYED_SENTINEL"] == 1
    assert rows[101]["CODE_GENDER"] is None
    assert rows[101]["FLAG_CODE_GENDER_XNA"] == 1
    assert rows[101]["ORGANIZATION_TYPE"] is None
    assert rows[101]["FLAG_ORGANIZATION_TYPE_XNA"] == 1
    assert rows[101]["TARGET"] == 1  # TARGET preserved

    # Row 102 was clean
    assert rows[102]["DAYS_EMPLOYED"] == -2500
    assert rows[102]["FLAG_DAYS_EMPLOYED_SENTINEL"] == 0
    assert rows[102]["CODE_GENDER"] == "F"
    assert rows[102]["FLAG_CODE_GENDER_XNA"] == 0
    assert rows[102]["ORGANIZATION_TYPE"] == "Business Entity Type 3"
    assert rows[102]["FLAG_ORGANIZATION_TYPE_XNA"] == 0
    assert rows[102]["TARGET"] == 0  # TARGET preserved


def test_synthetic_idempotency(spark: SparkSession):
    """Verify running cleaning multiple times produces identical results without duplicate columns."""
    schema = StructType([
        StructField("DAYS_EMPLOYED", IntegerType(), True),
        StructField("CODE_GENDER", StringType(), True),
    ])
    data = [(365243, "XNA")]
    df = spark.createDataFrame(data, schema)

    run1 = clean_application(df)
    run2 = clean_application(run1)

    assert run1.columns == run2.columns
    assert run1.collect() == run2.collect()


# ======================================================================
# Integration Tests: Bronze Inputs & Real Silver Outputs
# ======================================================================

def test_bronze_inputs_exist():
    """Verify all 8 Bronze directories exist and contain Parquet files."""
    for tbl in EXPECTED_TABLES:
        b_dir = BRONZE_BASE_DIR / tbl
        assert b_dir.exists(), f"Bronze directory missing: {b_dir}"
        parquet_files = list(b_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, f"No Parquet files found in Bronze directory: {b_dir}"


@require_silver
def test_silver_outputs_exist():
    """Verify all 8 Silver directories exist and contain Parquet files."""
    for tbl in EXPECTED_TABLES:
        s_dir = SILVER_BASE_DIR / tbl
        assert s_dir.exists(), f"Silver directory missing: {s_dir}"
        parquet_files = list(s_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, f"No Parquet files found in Silver directory: {s_dir}"


@require_silver
def test_silver_parquet_readable(spark: SparkSession):
    """Verify all 8 Silver Parquet tables can be read by Spark."""
    for tbl in EXPECTED_TABLES:
        s_dir = SILVER_BASE_DIR / tbl
        df = spark.read.parquet(str(s_dir))
        assert df.count() > 0, f"Silver table {tbl} is empty"


@require_silver
def test_bronze_silver_row_count_match(spark: SparkSession):
    """Verify Bronze row count equals Silver row count for every table."""
    for tbl in EXPECTED_TABLES:
        b_df = spark.read.parquet(str(BRONZE_BASE_DIR / tbl))
        s_df = spark.read.parquet(str(SILVER_BASE_DIR / tbl))
        b_cnt = b_df.count()
        s_cnt = s_df.count()
        assert b_cnt == s_cnt, f"Row count mismatch on {tbl}: Bronze={b_cnt}, Silver={s_cnt}"
        assert s_cnt == EXPECTED_ROW_COUNTS[tbl], f"Expected {EXPECTED_ROW_COUNTS[tbl]} on {tbl}, got {s_cnt}"


@require_silver
def test_application_train_days_employed_sentinel_zero(spark: SparkSession):
    """Verify DAYS_EMPLOYED == 365243 count becomes zero in application_train."""
    df = spark.read.parquet(str(SILVER_BASE_DIR / "application_train"))
    sentinel_cnt = df.filter(F.col("DAYS_EMPLOYED") == 365243).count()
    assert sentinel_cnt == 0, f"Found {sentinel_cnt} remaining sentinels in application_train.DAYS_EMPLOYED"


@require_silver
def test_application_test_days_employed_sentinel_zero(spark: SparkSession):
    """Verify DAYS_EMPLOYED == 365243 count becomes zero in application_test."""
    df = spark.read.parquet(str(SILVER_BASE_DIR / "application_test"))
    sentinel_cnt = df.filter(F.col("DAYS_EMPLOYED") == 365243).count()
    assert sentinel_cnt == 0, f"Found {sentinel_cnt} remaining sentinels in application_test.DAYS_EMPLOYED"


@require_silver
def test_sentinel_flags_equal_original_counts(spark: SparkSession):
    """Verify sentinel flags equal the Phase 1 documented counts."""
    # application_train: 55,374
    df_train = spark.read.parquet(str(SILVER_BASE_DIR / "application_train"))
    flag_train_cnt = df_train.filter(F.col("FLAG_DAYS_EMPLOYED_SENTINEL") == 1).count()
    assert flag_train_cnt == 55_374, f"Expected 55,374 sentinels in application_train, got {flag_train_cnt}"

    # application_test: 9,274
    df_test = spark.read.parquet(str(SILVER_BASE_DIR / "application_test"))
    flag_test_cnt = df_test.filter(F.col("FLAG_DAYS_EMPLOYED_SENTINEL") == 1).count()
    assert flag_test_cnt == 9_274, f"Expected 9,274 sentinels in application_test, got {flag_test_cnt}"


@require_silver
def test_previous_application_sentinels_cleaned(spark: SparkSession):
    """Verify all 5 previous_application sentinel columns have 0 sentinels and flags match expected counts."""
    expected_sentinels = {
        "DAYS_FIRST_DRAWING": 934_444,
        "DAYS_FIRST_DUE": 40_645,
        "DAYS_LAST_DUE_1ST_VERSION": 93_864,
        "DAYS_LAST_DUE": 211_221,
        "DAYS_TERMINATION": 225_913,
    }
    df = spark.read.parquet(str(SILVER_BASE_DIR / "previous_application"))

    for col_name, expected_cnt in expected_sentinels.items():
        sentinel_cnt = df.filter(F.col(col_name) == 365243).count()
        assert sentinel_cnt == 0, f"Column {col_name} still has {sentinel_cnt} sentinels"

        flag_col = f"FLAG_{col_name}_SENTINEL"
        flag_cnt = df.filter(F.col(flag_col) == 1).count()
        assert flag_cnt == expected_cnt, f"Flag {flag_col} count {flag_cnt} != expected {expected_cnt}"


@require_silver
def test_xna_values_removed_as_documented(spark: SparkSession):
    """Verify relevant XNA values are replaced with NULL in cleaned tables."""
    # 1. application_train
    df_train = spark.read.parquet(str(SILVER_BASE_DIR / "application_train"))
    assert df_train.filter(F.col("CODE_GENDER") == "XNA").count() == 0
    assert df_train.filter(F.col("ORGANIZATION_TYPE") == "XNA").count() == 0

    # 2. application_test
    df_test = spark.read.parquet(str(SILVER_BASE_DIR / "application_test"))
    assert df_test.filter(F.col("ORGANIZATION_TYPE") == "XNA").count() == 0

    # 3. previous_application
    df_prev = spark.read.parquet(str(SILVER_BASE_DIR / "previous_application"))
    for col_name in PREVIOUS_APPLICATION_XNA_COLUMNS:
        assert df_prev.filter(F.col(col_name) == "XNA").count() == 0, f"Column {col_name} has remaining XNA"

    # 4. POS_CASH_balance
    df_pos = spark.read.parquet(str(SILVER_BASE_DIR / "POS_CASH_balance"))
    assert df_pos.filter(F.col("NAME_CONTRACT_STATUS") == "XNA").count() == 0


@require_silver
def test_xna_flags_preserve_information(spark: SparkSession):
    """Verify XNA flags match original Phase 1 documented counts."""
    # application_train
    df_train = spark.read.parquet(str(SILVER_BASE_DIR / "application_train"))
    assert df_train.filter(F.col("FLAG_CODE_GENDER_XNA") == 1).count() == 4
    assert df_train.filter(F.col("FLAG_ORGANIZATION_TYPE_XNA") == 1).count() == 55_374

    # application_test
    df_test = spark.read.parquet(str(SILVER_BASE_DIR / "application_test"))
    assert df_test.filter(F.col("FLAG_ORGANIZATION_TYPE_XNA") == 1).count() == 9_274

    # POS_CASH_balance
    df_pos = spark.read.parquet(str(SILVER_BASE_DIR / "POS_CASH_balance"))
    assert df_pos.filter(F.col("FLAG_NAME_CONTRACT_STATUS_XNA") == 1).count() == 2


@require_silver
def test_target_values_unchanged(spark: SparkSession):
    """Verify TARGET column in application_train is unchanged between Bronze and Silver."""
    b_df = spark.read.parquet(str(BRONZE_BASE_DIR / "application_train"))
    s_df = spark.read.parquet(str(SILVER_BASE_DIR / "application_train"))

    b_targets = b_df.groupBy("TARGET").count().collect()
    s_targets = s_df.groupBy("TARGET").count().collect()

    b_map = {r["TARGET"]: r["count"] for r in b_targets}
    s_map = {r["TARGET"]: r["count"] for r in s_targets}

    assert b_map == s_map, f"TARGET distribution changed: {b_map} vs {s_map}"
    assert set(b_map.keys()) == {0, 1}


@require_silver
def test_negative_days_values_preserved(spark: SparkSession):
    """Verify negative DAYS_* values that are not sentinels remain negative and unchanged."""
    df = spark.read.parquet(str(SILVER_BASE_DIR / "application_train"))
    stats = df.select(
        F.min("DAYS_BIRTH").alias("min_birth"),
        F.max("DAYS_BIRTH").alias("max_birth"),
        F.count(F.when(F.col("DAYS_BIRTH") < 0, 1)).alias("neg_birth_cnt"),
    ).first().asDict()

    assert stats["min_birth"] == -25229
    assert stats["max_birth"] == -7489
    assert stats["neg_birth_cnt"] == 307_511


@require_silver
def test_no_generic_missing_value_imputation(spark: SparkSession):
    """Verify no generic missing value imputation occurred (nulls preserved)."""
    # RATE_INTEREST_PRIMARY in previous_application has > 99% nulls
    df_prev = spark.read.parquet(str(SILVER_BASE_DIR / "previous_application"))
    null_cnt = df_prev.filter(F.col("RATE_INTEREST_PRIMARY").isNull()).count()
    assert null_cnt == 1_664_263, f"Expected 1,664,263 nulls, got {null_cnt}"


@require_silver
def test_pos_cash_balance_row_count_unchanged(spark: SparkSession):
    """Verify POS_CASH_balance row count is exactly 10,001,358 (grain duplicates preserved)."""
    df = spark.read.parquet(str(SILVER_BASE_DIR / "POS_CASH_balance"))
    assert df.count() == 10_001_358


@require_silver
def test_installments_payments_row_count_unchanged(spark: SparkSession):
    """Verify installments_payments row count is exactly 13,605,401 (grain duplicates preserved)."""
    df = spark.read.parquet(str(SILVER_BASE_DIR / "installments_payments"))
    assert df.count() == 13_605_401


@require_silver
def test_no_unexpected_columns_removed(spark: SparkSession):
    """Verify no Bronze columns were dropped in Silver."""
    for tbl in EXPECTED_TABLES:
        b_df = spark.read.parquet(str(BRONZE_BASE_DIR / tbl))
        s_df = spark.read.parquet(str(SILVER_BASE_DIR / tbl))
        b_cols = set(b_df.columns)
        s_cols = set(s_df.columns)
        missing = b_cols - s_cols
        assert len(missing) == 0, f"Table {tbl} is missing Bronze columns in Silver: {missing}"


def test_raw_dataset_untouched():
    """Verify raw dataset/ folder and CSV files remain completely untouched."""
    assert DATASET_DIR.exists()
    csv_files = list(DATASET_DIR.glob("*.csv"))
    assert len(csv_files) >= 8

    # Verify key CSV files exist
    for tbl_name, tbl_cfg in TABLES.items():
        csv_path = pathlib.Path(tbl_cfg["path"])
        assert csv_path.exists(), f"Raw CSV missing: {csv_path}"
        assert csv_path.stat().st_size > 0, f"Raw CSV empty: {csv_path}"


@require_silver
def test_silver_reports_exist():
    """Verify all required Silver reports exist and are valid JSON/Markdown."""
    json_summary = REPORTS_DIR / "cleaning_summary.json"
    md_summary = REPORTS_DIR / "cleaning_summary.md"
    dq_json = REPORTS_DIR / "data_quality_before_after.json"

    assert json_summary.exists(), f"Missing {json_summary}"
    assert md_summary.exists(), f"Missing {md_summary}"
    assert dq_json.exists(), f"Missing {dq_json}"

    with open(json_summary, encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("all_passed") is True
    assert data.get("total_rows") == 58_489_893

    with open(dq_json, encoding="utf-8") as f:
        dq_data = json.load(f)
    assert "columns" in dq_data
    assert len(dq_data["columns"]) > 0
