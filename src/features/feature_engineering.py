"""Phase 5 Feature Engineering & Leakage Audit Layer.

Consumes Phase 4 Gold datasets (data/gold/model_input/ and data/gold/model_input_test/).
Applies defensible applicant-level derived features:
1. Validates Gold feature contract and invariants.
2. Constructs application-level and cross-table historical derived features.
3. Groups features into explicit feature families (ID, LABEL, APP, BURO, BURO_BAL, PREV, INST, CC, POS, DERIVED).
4. Conducts strict target and identifier leakage audits.
5. Verifies 100% train/test schema consistency.
6. Writes final model input Parquet datasets to data/model_input/ and data/model_input_test/.
7. Generates configs/features.yaml, reports/leakage_audit.*, and reports/feature_quality.*.
"""

from datetime import datetime, timezone
import json
import os
import pathlib
import sys
import time
from typing import Any

from pyspark.sql import Column, DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NumericType,
    StringType,
)
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.log import get_logger
from src.common.spark_session import get_spark

logger = get_logger("FeatureEngineering", "feature_engineering.log")

GOLD_TRAIN_DIR = ROOT / "data" / "gold" / "model_input"
GOLD_TEST_DIR = ROOT / "data" / "gold" / "model_input_test"

MODEL_INPUT_DIR = ROOT / "data" / "model_input"
MODEL_INPUT_TEST_DIR = ROOT / "data" / "model_input_test"

CONFIGS_DIR = ROOT / "configs"
REPORTS_DIR = ROOT / "reports"

SPINE_TRAIN_ROWS = 307_511
SPINE_TEST_ROWS = 48_744

SUSPICIOUS_KEYWORD_PATTERNS = [
    "TARGET",
    "DEFAULT",
    "OUTCOME",
    "LABEL",
    "RECOVERY",
    "LOSS",
    "WRITE_OFF",
    "COLLECTION",
    "CHARGEOFF",
]


def safe_divide(numerator: Column, denominator: Column) -> Column:
    """Safe division returning NULL when denominator is NULL, 0, or NaN."""
    return F.when(
        (denominator.isNotNull()) & (denominator != 0) & (~F.isnan(denominator)),
        numerator / denominator,
    ).otherwise(None)


def sanitize_numeric_column(col: Column) -> Column:
    """Replace NaN, inf, and -inf with NULL."""
    return F.when(
        F.isnan(col) | (col == float("inf")) | (col == float("-inf")),
        None,
    ).otherwise(col)


# ======================================================================
# 1. Gold Input Validation
# ======================================================================

def validate_gold_input(spark: SparkSession) -> tuple[DataFrame, DataFrame]:
    """Load and validate Phase 4 Gold training and inference datasets."""
    if not GOLD_TRAIN_DIR.exists():
        raise FileNotFoundError(f"Gold training input missing: {GOLD_TRAIN_DIR}")
    if not GOLD_TEST_DIR.exists():
        raise FileNotFoundError(f"Gold test input missing: {GOLD_TEST_DIR}")

    df_train = spark.read.parquet(str(GOLD_TRAIN_DIR))
    df_test = spark.read.parquet(str(GOLD_TEST_DIR))

    train_rows = df_train.count()
    test_rows = df_test.count()

    if train_rows != SPINE_TRAIN_ROWS:
        raise AssertionError(f"Gold train row count mismatch: expected {SPINE_TRAIN_ROWS:,}, got {train_rows:,}")
    if test_rows != SPINE_TEST_ROWS:
        raise AssertionError(f"Gold test row count mismatch: expected {SPINE_TEST_ROWS:,}, got {test_rows:,}")

    # Check key uniqueness
    train_distinct = df_train.select("SK_ID_CURR").distinct().count()
    test_distinct = df_test.select("SK_ID_CURR").distinct().count()
    if train_distinct != train_rows:
        raise AssertionError(f"Train SK_ID_CURR not unique: {train_distinct:,} distinct vs {train_rows:,} total")
    if test_distinct != test_rows:
        raise AssertionError(f"Test SK_ID_CURR not unique: {test_distinct:,} distinct vs {test_rows:,} total")

    # Check target
    if "TARGET" not in df_train.columns:
        raise AssertionError("TARGET missing in Gold training input")
    if "TARGET" in df_test.columns:
        raise AssertionError("TARGET illegally present in Gold test input")

    logger.info(f"Gold inputs validated: Train={train_rows:,} rows, Test={test_rows:,} rows")
    return df_train, df_test


# ======================================================================
# 2. Derived Feature Engineering
# ======================================================================

def build_application_derived_features(df: DataFrame, raw_app_cols: list[str]) -> DataFrame:
    """Engineer defensible applicant-level ratios and missingness indicators."""
    # 1. Income-to-Credit & Dependency Ratios
    if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
        df = df.withColumn(
            "APP_INCOME_TO_CREDIT_RATIO",
            safe_divide(F.col("AMT_INCOME_TOTAL"), F.col("AMT_CREDIT")),
        )
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df = df.withColumn(
            "APP_ANNUITY_TO_INCOME_PERCENT",
            safe_divide(F.col("AMT_ANNUITY") * F.lit(12.0), F.col("AMT_INCOME_TOTAL")),
        )
    if "CNT_CHILDREN" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df = df.withColumn(
            "APP_CHILDREN_TO_FAMILY_RATIO",
            safe_divide(F.col("CNT_CHILDREN"), F.col("CNT_FAM_MEMBERS")),
        )
    if "AMT_CREDIT" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df = df.withColumn(
            "APP_CREDIT_PER_PERSON",
            safe_divide(F.col("AMT_CREDIT"), F.col("CNT_FAM_MEMBERS")),
        )
    if "AMT_INCOME_TOTAL" in df.columns and "CNT_CHILDREN" in df.columns:
        df = df.withColumn(
            "APP_INCOME_PER_CHILD",
            safe_divide(F.col("AMT_INCOME_TOTAL"), F.col("CNT_CHILDREN")),
        )

    # 2. External Sources Agreement / Dispersion
    # Calculate min, max, range across EXT_SOURCE_1, 2, 3
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if len(ext_cols) == 3:
        df = df.withColumn(
            "APP_EXT_SOURCES_MIN",
            F.least(F.col("EXT_SOURCE_1"), F.col("EXT_SOURCE_2"), F.col("EXT_SOURCE_3")),
        ).withColumn(
            "APP_EXT_SOURCES_MAX",
            F.greatest(F.col("EXT_SOURCE_1"), F.col("EXT_SOURCE_2"), F.col("EXT_SOURCE_3")),
        ).withColumn(
            "APP_EXT_SOURCES_RANGE",
            F.greatest(F.col("EXT_SOURCE_1"), F.col("EXT_SOURCE_2"), F.col("EXT_SOURCE_3")) -
            F.least(F.col("EXT_SOURCE_1"), F.col("EXT_SOURCE_2"), F.col("EXT_SOURCE_3")),
        )

    # 3. Missingness Features across baseline application fields (strictly excluding SK_ID_CURR and TARGET)
    if raw_app_cols:
        eval_cols = [c for c in raw_app_cols if c in df.columns]
        if eval_cols:
            missing_expr = F.lit(0)
            for col_name in eval_cols:
                missing_expr = missing_expr + F.when(F.col(col_name).isNull(), 1).otherwise(0)
            df = df.withColumn("APP_MISSING_COUNT", missing_expr)
            total_eval_cols = float(len(eval_cols))
            df = df.withColumn("APP_MISSING_RATIO", F.col("APP_MISSING_COUNT") / F.lit(total_eval_cols))

    return df


def build_historical_derived_features(df: DataFrame) -> DataFrame:
    """Engineer cross-table historical behavior and leverage ratios from existing Gold aggregates."""
    # 1. Total Debt to Income Ratio (Bureau Debt + Current Credit / Total Income)
    if "BURO_AMT_CREDIT_SUM_DEBT_SUM" in df.columns and "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        total_debt = F.coalesce(F.col("BURO_AMT_CREDIT_SUM_DEBT_SUM"), F.lit(0.0)) + F.coalesce(F.col("AMT_CREDIT"), F.lit(0.0))
        df = df.withColumn(
            "DERIVED_TOTAL_DEBT_TO_INCOME_RATIO",
            safe_divide(total_debt, F.col("AMT_INCOME_TOTAL")),
        )

    # 2. Bureau to Current Credit Scale Ratio
    if "BURO_AMT_CREDIT_SUM_SUM" in df.columns and "AMT_CREDIT" in df.columns:
        df = df.withColumn(
            "DERIVED_BURO_TO_CURRENT_CREDIT_RATIO",
            safe_divide(F.col("BURO_AMT_CREDIT_SUM_SUM"), F.col("AMT_CREDIT")),
        )

    # 3. Previous Approved Credit to Current Credit Ratio
    if "PREV_AMT_CREDIT_SUM" in df.columns and "AMT_CREDIT" in df.columns:
        df = df.withColumn(
            "DERIVED_PREV_APPROVED_TO_CURRENT_CREDIT_RATIO",
            safe_divide(F.col("PREV_AMT_CREDIT_SUM"), F.col("AMT_CREDIT")),
        )

    # 4. Overall Historical Delinquency Flag
    delinq_cols = [c for c in ["BURO_AMT_CREDIT_SUM_OVERDUE_SUM", "INST_LATE_COUNT", "CC_DPD_MONTH_COUNT", "POS_DPD_MONTH_COUNT"] if c in df.columns]
    if delinq_cols:
        conditions = [(F.coalesce(F.col(c), F.lit(0)) > 0) for c in delinq_cols]
        has_overdue = conditions[0]
        for cond in conditions[1:]:
            has_overdue = has_overdue | cond
        df = df.withColumn(
            "DERIVED_OVERALL_OVERDUE_FLAG",
            F.when(has_overdue, 1).otherwise(0).cast(IntegerType()),
        )

    # 5. Payment Discipline Score (Installment Payment Ratio * (1 - Late Rate))
    if "INST_PAYMENT_RATIO" in df.columns and "INST_LATE_RATE" in df.columns:
        disp_condition = (F.col("INST_PAYMENT_RATIO").isNotNull()) & (F.col("INST_LATE_RATE").isNotNull())
        df = df.withColumn(
            "DERIVED_PAYMENT_DISCIPLINE_SCORE",
            F.when(disp_condition, F.col("INST_PAYMENT_RATIO") * (F.lit(1.0) - F.col("INST_LATE_RATE"))).otherwise(None),
        )

    # 6. Total Historical Installment Paid to Current Income Ratio
    if "INST_AMT_PAYMENT_SUM" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df = df.withColumn(
            "DERIVED_TOTAL_PAID_TO_INCOME_RATIO",
            safe_divide(F.col("INST_AMT_PAYMENT_SUM"), F.col("AMT_INCOME_TOTAL")),
        )

    # 7. Revolving Card Balance Burden to Income Ratio
    if "CC_AMT_BALANCE_MEAN" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df = df.withColumn(
            "DERIVED_CARD_UTILIZATION_BURDEN",
            safe_divide(F.col("CC_AMT_BALANCE_MEAN"), F.col("AMT_INCOME_TOTAL")),
        )

    # Sanitize all newly added derived columns to guarantee zero NaN/Inf
    derived_cols = [
        "DERIVED_TOTAL_DEBT_TO_INCOME_RATIO",
        "DERIVED_BURO_TO_CURRENT_CREDIT_RATIO",
        "DERIVED_PREV_APPROVED_TO_CURRENT_CREDIT_RATIO",
        "DERIVED_PAYMENT_DISCIPLINE_SCORE",
        "DERIVED_TOTAL_PAID_TO_INCOME_RATIO",
        "DERIVED_CARD_UTILIZATION_BURDEN",
    ]
    for c in derived_cols:
        if c in df.columns:
            df = df.withColumn(c, sanitize_numeric_column(F.col(c)))

    return df


def get_raw_application_columns(gold_cols: list[str]) -> list[str]:
    """Identify baseline application columns for missingness counts (excluding ID, TARGET, and aggregates)."""
    exclude_prefixes = ("APP_", "BURO_", "PREV_", "INST_", "CC_", "POS_", "DERIVED_")
    raw_cols = []
    for c in gold_cols:
        if c in ("SK_ID_CURR", "TARGET"):
            continue
        if any(c.startswith(prefix) for prefix in exclude_prefixes):
            continue
        raw_cols.append(c)
    return raw_cols


# ======================================================================
# 3. Train / Test Consistency & Leakage Audits
# ======================================================================

def validate_feature_schema(train_df: DataFrame, test_df: DataFrame) -> dict[str, Any]:
    """Verify that train and test have identical schemas (except TARGET in train)."""
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols

    schema_match = (missing_in_test == {"TARGET"}) and (len(missing_in_train) == 0)

    train_types = {f.name: str(f.dataType) for f in train_df.schema.fields}
    test_types = {f.name: str(f.dataType) for f in test_df.schema.fields}

    type_mismatches = []
    for col in test_cols:
        if col in train_types and train_types[col] != test_types[col]:
            type_mismatches.append({
                "column": col,
                "train_type": train_types[col],
                "test_type": test_types[col],
            })

    type_match = len(type_mismatches) == 0

    return {
        "schema_match": schema_match,
        "type_match": type_match,
        "missing_in_test": list(missing_in_test),
        "missing_in_train": list(missing_in_train),
        "type_mismatches": type_mismatches,
        "train_feature_count": len(train_cols),
        "test_feature_count": len(test_cols),
    }


def audit_leakage(train_df: DataFrame, test_df: DataFrame) -> dict[str, Any]:
    """Perform formal leakage audit checking expressions, names, and identifiers."""
    train_cols = train_df.columns
    test_cols = test_df.columns

    # 1. Verify TARGET presence / absence
    target_in_train = "TARGET" in train_cols
    target_in_test = "TARGET" in test_cols
    target_rule_pass = target_in_train and not target_in_test

    # 2. Audit suspicious keywords in column names
    suspicious_findings: list[dict[str, str]] = []
    for col in train_cols:
        col_upper = col.upper()
        for pattern in SUSPICIOUS_KEYWORD_PATTERNS:
            if pattern in col_upper:
                # Classify
                if col == "TARGET":
                    classification = "EXCLUDED_LABEL"
                    reason = "Supervised target label, excluded from predictive model matrix"
                elif "SK_DPD_DEF" in col:
                    classification = "SAFE"
                    reason = "Historical days past due tolerance metric from past contracts; not current loan label"
                elif "DEF_30_CNT_SOCIAL_CIRCLE" in col or "DEF_60_CNT_SOCIAL_CIRCLE" in col:
                    classification = "SAFE"
                    reason = "Social circle historical inquiry attribute; not current loan label"
                elif "CODE_REJECT_REASON" in col:
                    classification = "SAFE"
                    reason = "Historical rejection reason on previous applications; not current loan label"
                else:
                    classification = "SUSPICIOUS"
                    reason = f"Contains pattern '{pattern}'"
                suspicious_findings.append({
                    "column": col,
                    "pattern": pattern,
                    "classification": classification,
                    "reason": reason,
                })

    # 3. Verify SK_ID_CURR handling
    id_in_train = "SK_ID_CURR" in train_cols
    id_in_test = "SK_ID_CURR" in test_cols
    id_rule_pass = id_in_train and id_in_test

    # 4. Check for illegal target-derived expressions in column names
    target_derived_cols = [c for c in train_cols if c != "TARGET" and "TARGET" in c.upper()]
    target_derived_pass = len(target_derived_cols) == 0

    overall_leakage_pass = target_rule_pass and id_rule_pass and target_derived_pass

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_leakage_status": "PASS" if overall_leakage_pass else "FAIL",
        "target_in_train": target_in_train,
        "target_in_test": target_in_test,
        "target_rule_pass": target_rule_pass,
        "id_rule_pass": id_rule_pass,
        "target_derived_columns": target_derived_cols,
        "target_derived_pass": target_derived_pass,
        "suspicious_features_evaluated": suspicious_findings,
    }


def audit_feature_quality(df: DataFrame, table_name: str) -> dict[str, Any]:
    """Calculate summary diagnostics and verify absence of Infinity / NaN."""
    total_rows = df.count()
    numeric_fields = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, (NumericType, IntegerType, LongType, DoubleType, FloatType))
    ]

    # Aggregate invalid values (NaN, Inf, -Inf) across numeric fields
    invalid_checks = []
    for c in numeric_fields:
        invalid_checks.append(
            F.count(F.when(F.isnan(F.col(c)) | (F.col(c) == float("inf")) | (F.col(c) == float("-inf")), 1)).alias(c)
        )

    # Batch invalid checks in chunks of 50 columns to optimize Catalyst execution
    invalid_counts: dict[str, int] = {}
    chunk_size = 50
    for i in range(0, len(invalid_checks), chunk_size):
        chunk = invalid_checks[i : i + chunk_size]
        res = df.select(chunk).first().asDict()
        for k, v in res.items():
            if v and v > 0:
                invalid_counts[k] = int(v)

    has_invalid = len(invalid_counts) > 0

    return {
        "table": table_name,
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "numeric_columns_count": len(numeric_fields),
        "invalid_numeric_columns": invalid_counts,
        "has_invalid_values": has_invalid,
        "status": "FAIL" if has_invalid else "PASS",
    }


# ======================================================================
# 4. Feature Registry Generation
# ======================================================================

def update_feature_registry(df: DataFrame) -> dict[str, Any]:
    """Generate complete, updated configs/features.yaml metadata dictionary."""
    registry: dict[str, Any] = {}

    for c in df.columns:
        if c == "SK_ID_CURR":
            registry[c] = {
                "family": "ID",
                "source": "application",
                "source_columns": ["SK_ID_CURR"],
                "definition": "Primary applicant loan identifier",
                "description": "Unique identifier for customer loan application",
                "dtype": "int",
                "null_behavior": "never_null",
                "leakage_status": "EXCLUDED_ID",
                "excluded_from_model": True,
            }
        elif c == "TARGET":
            registry[c] = {
                "family": "LABEL",
                "source": "application_train",
                "source_columns": ["TARGET"],
                "definition": "Supervised binary classification target (1 = default, 0 = repaid)",
                "description": "Loan default outcome label",
                "dtype": "int",
                "null_behavior": "never_null",
                "leakage_status": "EXCLUDED_LABEL",
                "excluded_from_model": True,
            }
        elif c.startswith("DERIVED_"):
            registry[c] = {
                "family": "DERIVED",
                "source": "cross_domain",
                "source_columns": ["multiple_gold_aggregates"],
                "definition": f"Cross-domain derived risk ratio: {c}",
                "description": f"Engineered leverage/behavioral feature {c}",
                "dtype": "float",
                "null_behavior": "null_on_zero_or_missing",
                "leakage_status": "SAFE",
                "excluded_from_model": False,
            }
        elif c.startswith("APP_"):
            registry[c] = {
                "family": "APP",
                "source": "application",
                "source_columns": ["application_attributes"],
                "definition": f"Application-level domain ratio: {c}",
                "description": f"Engineered application ratio {c}",
                "dtype": "float",
                "null_behavior": "safe_division_null_on_zero",
                "leakage_status": "SAFE",
                "excluded_from_model": False,
            }
        elif c.startswith("BURO_BAL_"):
            registry[c] = {
                "family": "BURO_BAL",
                "source": "bureau_balance",
                "source_columns": ["STATUS", "MONTHS_BALANCE"],
                "definition": f"bureau_balance aggregate: {c}",
                "description": f"Credit bureau monthly balance aggregate feature: {c}",
                "dtype": "numeric",
                "null_behavior": "null_if_no_bureau_balance",
                "leakage_status": "SAFE",
                "excluded_from_model": False,
            }
        elif c.startswith("BURO_"):
            registry[c] = {
                "family": "BURO",
                "source": "bureau",
                "source_columns": ["bureau attributes"],
                "definition": f"bureau aggregate: {c}",
                "description": f"Credit bureau historical account aggregate feature: {c}",
                "dtype": "numeric",
                "null_behavior": "null_if_no_bureau",
                "leakage_status": "SAFE",
                "excluded_from_model": False,
            }
        elif c.startswith("PREV_"):
            registry[c] = {
                "family": "PREV",
                "source": "previous_application",
                "source_columns": ["previous_application attributes"],
                "definition": f"previous_application aggregate: {c}",
                "description": f"Previous Home Credit application aggregate feature: {c}",
                "dtype": "numeric",
                "null_behavior": "null_if_no_prev_app",
                "leakage_status": "SAFE",
                "excluded_from_model": False,
            }
        elif c.startswith("INST_"):
            registry[c] = {
                "family": "INST",
                "source": "installments_payments",
                "source_columns": ["installments attributes"],
                "definition": f"installments_payments aggregate: {c}",
                "description": f"Installment payment behavioral aggregate feature: {c}",
                "dtype": "numeric",
                "null_behavior": "null_if_no_installments",
                "leakage_status": "SAFE",
                "excluded_from_model": False,
            }
        elif c.startswith("CC_"):
            registry[c] = {
                "family": "CC",
                "source": "credit_card_balance",
                "source_columns": ["credit_card_balance attributes"],
                "definition": f"credit_card_balance aggregate: {c}",
                "description": f"Credit card revolving balance aggregate feature: {c}",
                "dtype": "numeric",
                "null_behavior": "null_if_no_credit_card",
                "leakage_status": "SAFE",
                "excluded_from_model": False,
            }
        elif c.startswith("POS_"):
            registry[c] = {
                "family": "POS",
                "source": "POS_CASH_balance",
                "source_columns": ["POS_CASH_balance attributes"],
                "definition": f"POS_CASH_balance aggregate: {c}",
                "description": f"POS and cash loan monthly behavioral feature: {c}",
                "dtype": "numeric",
                "null_behavior": "null_if_no_pos_cash",
                "leakage_status": "SAFE",
                "excluded_from_model": False,
            }
        else:
            registry[c] = {
                "family": "APP",
                "source": "application",
                "source_columns": [c],
                "definition": f"Baseline application feature: {c}",
                "description": f"Cleaned raw application attribute: {c}",
                "dtype": "original",
                "null_behavior": "preserve_raw_missingness",
                "leakage_status": "SAFE",
                "excluded_from_model": False,
            }

    return registry


# ======================================================================
# 5. Full Pipeline Execution
# ======================================================================

def build_model_input(spark: SparkSession | None = None) -> dict[str, Any]:
    """Execute end-to-end Phase 5 Feature Engineering & Leakage Audit pipeline."""
    should_stop_spark = False
    if spark is None:
        spark = get_spark("LoanDefaultPrediction-Phase5-Features")
        should_stop_spark = True

    start_time = time.time()
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_INPUT_TEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load and Validate Gold inputs
        print("[1/6] Loading and validating Phase 4 Gold inputs ...", flush=True)
        df_train_gold, df_test_gold = validate_gold_input(spark)

        # Step 2: Determine baseline application columns for missingness
        raw_app_cols = get_raw_application_columns(df_train_gold.columns)
        logger.info(f"Identified {len(raw_app_cols)} baseline application columns for missingness features.")

        # Step 3: Engineer Derived Features on Train and Test
        print("[2/6] Engineering derived application & historical features ...", flush=True)
        df_train_fe = build_application_derived_features(df_train_gold, raw_app_cols)
        df_train_fe = build_historical_derived_features(df_train_fe)

        df_test_fe = build_application_derived_features(df_test_gold, raw_app_cols)
        df_test_fe = build_historical_derived_features(df_test_fe)

        # Step 4: Validate Train/Test Schema Consistency
        print("[3/6] Validating train/test schema consistency ...", flush=True)
        schema_audit = validate_feature_schema(df_train_fe, df_test_fe)
        if not schema_audit["schema_match"]:
            raise AssertionError(f"Train/Test schema mismatch: {schema_audit}")

        # Step 5: Leakage Audit
        print("[4/6] Conducting formal target & identifier leakage audit ...", flush=True)
        leakage_audit_data = audit_leakage(df_train_fe, df_test_fe)
        if leakage_audit_data["overall_leakage_status"] != "PASS":
            raise AssertionError(f"Target leakage detected! {leakage_audit_data}")

        # Step 6: Write Final Model Input Parquet Datasets
        print("[5/6] Writing final model input Parquet datasets ...", flush=True)
        # Train output: write to data/model_input/model_input.parquet
        train_out_path = str(MODEL_INPUT_DIR / "model_input.parquet")
        df_train_fe.write.mode("overwrite").parquet(train_out_path)
        df_train_readback = spark.read.parquet(train_out_path)

        # Test output: write to data/model_input_test/model_input_test.parquet
        test_out_path = str(MODEL_INPUT_TEST_DIR / "model_input_test.parquet")
        df_test_fe.write.mode("overwrite").parquet(test_out_path)
        df_test_readback = spark.read.parquet(test_out_path)

        # Post-write assertions
        if df_train_readback.count() != SPINE_TRAIN_ROWS:
            raise AssertionError("Train readback row count mismatch!")
        if df_test_readback.count() != SPINE_TEST_ROWS:
            raise AssertionError("Test readback row count mismatch!")

        # Step 7: Feature Quality Audit on Final Datasets
        print("[6/6] Auditing feature quality and generating reports ...", flush=True)
        train_dq = audit_feature_quality(df_train_readback, "model_input_train")
        test_dq = audit_feature_quality(df_test_readback, "model_input_test")

        if train_dq["has_invalid_values"] or test_dq["has_invalid_values"]:
            raise AssertionError(
                f"Invalid numerical values (NaN/Inf) detected in final model input! "
                f"Train: {train_dq['invalid_numeric_columns']}, Test: {test_dq['invalid_numeric_columns']}"
            )

        # Generate Feature Registry YAML
        feature_registry = update_feature_registry(df_train_readback)
        features_yaml_path = CONFIGS_DIR / "features.yaml"
        with open(features_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(feature_registry, f, sort_keys=False, indent=2)
        logger.info(f"Updated feature registry: {features_yaml_path}")

        # Write Reports
        total_runtime = time.time() - start_time

        def get_dir_size_mb(path: pathlib.Path) -> float:
            total = sum(f.stat().st_size for f in path.rglob("*.parquet") if f.is_file())
            return round(total / (1024 * 1024), 2)

        summary_data = {
            "project": "Loan Default Prediction Using Big Data Analytics",
            "phase": "Phase 5 — Feature Engineering & Leakage Audit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "train_rows": SPINE_TRAIN_ROWS,
            "train_columns": len(df_train_readback.columns),
            "test_rows": SPINE_TEST_ROWS,
            "test_columns": len(df_test_readback.columns),
            "train_size_mb": get_dir_size_mb(MODEL_INPUT_DIR),
            "test_size_mb": get_dir_size_mb(MODEL_INPUT_TEST_DIR),
            "total_runtime_seconds": round(total_runtime, 2),
            "schema_audit": schema_audit,
            "leakage_audit": leakage_audit_data,
            "train_data_quality": train_dq,
            "test_data_quality": test_dq,
            "all_passed": True,
        }

        write_phase5_reports(summary_data)
        return summary_data

    finally:
        if should_stop_spark:
            spark.stop()


def write_phase5_reports(summary: dict[str, Any]) -> None:
    """Generate reports/leakage_audit.* and reports/feature_quality.*."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. reports/leakage_audit.json
    leakage_json_path = REPORTS_DIR / "leakage_audit.json"
    with open(leakage_json_path, "w", encoding="utf-8") as f:
        json.dump(summary["leakage_audit"], f, indent=2)
    logger.info(f"Written leakage audit JSON: {leakage_json_path}")

    # 2. reports/leakage_audit.md
    leakage_md_path = REPORTS_DIR / "leakage_audit.md"
    l_data = summary["leakage_audit"]
    l_lines = [
        "# Phase 5 Target & Identifier Leakage Audit Report",
        "",
        "> **Project:** Loan Default Prediction Using Big Data Analytics  ",
        f"> **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"> **Audit Status:** **{l_data['overall_leakage_status']}**  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "This audit ensures that no target leakage, identifier correlation, or post-outcome variables enter the model feature matrix:",
        "",
        f"- **Target Present in Train:** {'YES (Valid Ground Truth)' if l_data['target_in_train'] else 'FAIL'}  ",
        f"- **Target Present in Test:** {'NO (Strictly Excluded)' if not l_data['target_in_test'] else 'FAIL'}  ",
        f"- **Target-Derived Columns:** {len(l_data['target_derived_columns'])} (Zero illegal target expressions)  ",
        f"- **Identifier Retention:** `SK_ID_CURR` retained as tracking ID but explicitly marked `excluded_from_model: true` in registry.  ",
        "",
        "---",
        "",
        "## 2. Suspicious Keyword Audit Results",
        "",
        "All features matching suspicious substring patterns (`TARGET`, `DEFAULT`, `OUTCOME`, `LABEL`, `RECOVERY`, `LOSS`, `WRITE_OFF`, `COLLECTION`, `CHARGEOFF`) were systematically evaluated:",
        "",
        "| Feature Name | Keyword | Classification | Audit Rationale |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for item in l_data["suspicious_features_evaluated"]:
        l_lines.append(
            f"| `{item['column']}` | `{item['pattern']}` | **{item['classification']}** | {item['reason']} |"
        )

    l_lines.extend([
        "",
        "---",
        "",
        "## 3. Train / Test Consistency Audit",
        "",
        f"- **Train Feature Count:** {summary['schema_audit']['train_feature_count']} features (including `TARGET`)  ",
        f"- **Test Feature Count:** {summary['schema_audit']['test_feature_count']} features (strictly NO `TARGET`)  ",
        f"- **Exact Schema Parity:** {'PASS' if summary['schema_audit']['schema_match'] else 'FAIL'}  ",
        f"- **Data Type Parity:** {'PASS' if summary['schema_audit']['type_match'] else 'FAIL'}  ",
        "",
    ])

    with open(leakage_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(l_lines))
    logger.info(f"Written leakage audit Markdown: {leakage_md_path}")

    # 3. reports/feature_quality.json
    fq_json_path = REPORTS_DIR / "feature_quality.json"
    fq_data = {
        "timestamp": summary["timestamp"],
        "train_rows": summary["train_rows"],
        "train_columns": summary["train_columns"],
        "test_rows": summary["test_rows"],
        "test_columns": summary["test_columns"],
        "train_invalid_numerics": summary["train_data_quality"]["invalid_numeric_columns"],
        "test_invalid_numerics": summary["test_data_quality"]["invalid_numeric_columns"],
        "status": "PASS",
    }
    with open(fq_json_path, "w", encoding="utf-8") as f:
        json.dump(fq_data, f, indent=2)
    logger.info(f"Written feature quality JSON: {fq_json_path}")

    # 4. reports/feature_quality.md
    fq_md_path = REPORTS_DIR / "feature_quality.md"
    fq_lines = [
        "# Phase 5 Feature Quality & Data Sanity Report",
        "",
        "> **Project:** Loan Default Prediction Using Big Data Analytics  ",
        f"> **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"> **Training Rows:** {summary['train_rows']:,}  ",
        f"> **Inference Rows:** {summary['test_rows']:,}  ",
        f"> **Training Features:** {summary['train_columns']}  ",
        f"> **Inference Features:** {summary['test_columns']}  ",
        "",
        "---",
        "",
        "## 1. Numeric Sanity & Safe Division",
        "",
        "- **Infinity / -Infinity Count:** 0 (Zero occurrences)  ",
        "- **NaN Count:** 0 (Zero occurrences)  ",
        "- **Safe Division Policy:** All ratio calculations return `NULL` when denominator is 0, NULL, or NaN.  ",
        "",
        "---",
        "",
        "## 2. Dataset Outputs",
        "",
        f"- **Final Training Parquet:** `data/model_input/model_input.parquet` ({summary['train_size_mb']:.2f} MB)  ",
        f"- **Final Inference Parquet:** `data/model_input_test/model_input_test.parquet` ({summary['test_size_mb']:.2f} MB)  ",
        "- **Feature Registry:** [`configs/features.yaml`](../../configs/features.yaml)  ",
        "",
    ]
    with open(fq_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fq_lines))
    logger.info(f"Written feature quality Markdown: {fq_md_path}")
