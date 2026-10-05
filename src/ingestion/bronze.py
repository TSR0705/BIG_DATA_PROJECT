"""Phase 2 Bronze Ingestion Layer.

Converts raw Home Credit CSV files from dataset/ into typed, validated Parquet
tables in data/bronze/ using frozen Phase 1 StructType schemas.
Preserves raw data semantically without cleaning or transformation.
"""

from datetime import datetime, timezone
import json
import os
import pathlib
import sys
import time
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.schemas import TABLE_SCHEMAS, validate_schema_contract
from configs.tables import TABLES
from src.common.log import get_logger
from src.common.spark_session import get_spark

logger = get_logger("BronzeIngestion", "bronze_ingestion.log")

BRONZE_BASE_DIR = ROOT / "data" / "bronze"
REPORTS_DIR = ROOT / "reports" / "bronze"

# Sensible partition counts for 16 GB laptop in local[*] mode
TABLE_PARTITIONS: dict[str, int] = {
    "application_train": 2,
    "application_test": 1,
    "bureau": 4,
    "bureau_balance": 32,
    "previous_application": 4,
    "installments_payments": 16,
    "credit_card_balance": 8,
    "POS_CASH_balance": 12,
}

# Verified Phase 1 reference counts as sanity check
PHASE1_EXPECTED_COUNTS: dict[str, int] = {
    "application_train": 307_511,
    "application_test": 48_744,
    "bureau": 1_716_428,
    "bureau_balance": 27_299_925,
    "previous_application": 1_670_214,
    "installments_payments": 13_605_401,
    "credit_card_balance": 3_840_312,
    "POS_CASH_balance": 10_001_358,
}


def get_bronze_dir(table_name: str) -> pathlib.Path:
    """Return the filesystem directory path for a Bronze table."""
    return BRONZE_BASE_DIR / table_name


def get_bronze_path(table_name: str) -> str:
    """Return the string path for a Bronze table."""
    return str(get_bronze_dir(table_name))


def validate_bronze_schema(
    bronze_schema: StructType,
    expected_schema: StructType,
    table_name: str,
) -> tuple[bool, list[str]]:
    """Validate that Bronze Parquet schema matches expected schema in names, types, and order."""
    errors = []
    bronze_fields = bronze_schema.fields
    expected_fields = expected_schema.fields

    if len(bronze_fields) != len(expected_fields):
        errors.append(
            f"Field count mismatch: expected {len(expected_fields)}, got {len(bronze_fields)}"
        )

    for idx, (b_field, e_field) in enumerate(zip(bronze_fields, expected_fields)):
        if b_field.name != e_field.name:
            errors.append(
                f"Field #{idx} name mismatch: expected '{e_field.name}', got '{b_field.name}'"
            )
        if b_field.dataType != e_field.dataType:
            errors.append(
                f"Field '{e_field.name}' type mismatch: expected {e_field.dataType}, got {b_field.dataType}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def get_dir_size_and_file_count(directory: pathlib.Path) -> tuple[int, int]:
    """Calculate total byte size and count of Parquet part files in a directory."""
    total_bytes = 0
    file_count = 0
    if directory.exists():
        for file in directory.glob("*.parquet"):
            if file.is_file():
                total_bytes += file.stat().st_size
                file_count += 1
    return total_bytes, file_count


def ingest_table(
    name: str,
    spark: SparkSession | None = None,
) -> dict[str, Any]:
    """Ingest a single Home Credit CSV into typed Bronze Parquet with strict validation."""
    if name not in TABLES:
        raise KeyError(f"Table '{name}' not found in configs/tables.py registry")
    if name not in TABLE_SCHEMAS:
        raise KeyError(f"Table '{name}' not found in configs/schemas.py")

    should_stop_spark = False
    if spark is None:
        spark = get_spark("LoanDefaultPrediction-BronzeIngestion")
        should_stop_spark = True

    tbl_cfg = TABLES[name]
    csv_path = tbl_cfg["path"]
    expected_schema = TABLE_SCHEMAS[name]
    bronze_dir = get_bronze_dir(name)
    bronze_path = str(bronze_dir)
    target_partitions = TABLE_PARTITIONS.get(name, 4)

    logger.info(f"[{name}] Starting Bronze ingestion from {csv_path} -> {bronze_path}")
    start_time = time.time()

    try:
        # Step 1: Read raw CSV with explicit schema and inferSchema=False
        df_csv: DataFrame = (
            spark.read
            .option("header", "true")
            .option("quote", "\"")
            .option("escape", "\"")
            .schema(expected_schema)
            .csv(csv_path)
        )

        # Dynamic row count on raw CSV with schema applied
        source_count = df_csv.count()
        source_cols = len(df_csv.columns)
        logger.info(f"[{name}] Source CSV loaded: {source_count:,} rows, {source_cols} columns")

        # Sanity check against Phase 1 empirical count
        expected_ref = PHASE1_EXPECTED_COUNTS.get(name)
        if expected_ref is not None and source_count != expected_ref:
            logger.warning(
                f"[{name}] Source count {source_count:,} differs from Phase 1 count {expected_ref:,}"
            )

        # Step 2: Sensible partitioning
        bronze_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_partitions == 1:
            df_to_write = df_csv.coalesce(1)
        elif target_partitions > 1:
            df_to_write = df_csv.repartition(target_partitions)
        else:
            df_to_write = df_csv

        # Step 3: Write Parquet with overwrite mode (idempotent)
        t_write_start = time.time()
        (
            df_to_write.write
            .mode("overwrite")
            .option("compression", "snappy")
            .parquet(bronze_path)
        )
        write_duration = time.time() - t_write_start
        logger.info(f"[{name}] Parquet write completed in {write_duration:.2f}s")

        # Step 4: Parquet read-back validation
        t_val_start = time.time()
        bronze_df: DataFrame = spark.read.parquet(bronze_path)
        bronze_count = bronze_df.count()
        bronze_cols = len(bronze_df.columns)
        validation_duration = time.time() - t_val_start

        # Row-count assertion
        if bronze_count != source_count:
            raise AssertionError(
                f"[{name}] Row count mismatch! Source CSV has {source_count:,} rows, "
                f"but Bronze Parquet has {bronze_count:,} rows."
            )

        # Schema assertion
        schema_valid, schema_errors = validate_bronze_schema(
            bronze_schema=bronze_df.schema,
            expected_schema=expected_schema,
            table_name=name,
        )
        if not schema_valid:
            error_msg = "; ".join(schema_errors)
            raise AssertionError(f"[{name}] Bronze schema validation failed: {error_msg}")

        # Metrics & sizing
        total_bytes, file_count = get_dir_size_and_file_count(bronze_dir)
        total_duration = time.time() - start_time

        result: dict[str, Any] = {
            "table": name,
            "source_path": csv_path,
            "bronze_path": bronze_path,
            "source_rows": source_count,
            "bronze_rows": bronze_count,
            "row_count_match": True,
            "source_columns": source_cols,
            "bronze_columns": bronze_cols,
            "schema_match": True,
            "target_partitions": target_partitions,
            "output_file_count": file_count,
            "output_size_bytes": total_bytes,
            "output_size_mb": round(total_bytes / (1024 * 1024), 2),
            "write_duration_seconds": round(write_duration, 2),
            "validation_duration_seconds": round(validation_duration, 2),
            "total_duration_seconds": round(total_duration, 2),
            "status": "PASS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"[{name}] Ingestion PASS: {bronze_count:,} rows | "
            f"{result['output_size_mb']} MB ({file_count} files) | "
            f"Total time: {total_duration:.2f}s"
        )
        return result

    except Exception as exc:
        logger.error(f"[{name}] Ingestion FAILED: {exc}", exc_info=True)
        raise

    finally:
        if should_stop_spark:
            spark.stop()


def ingest_all(spark: SparkSession | None = None) -> dict[str, Any]:
    """Ingest all 8 Home Credit tables into Bronze layer sequentially."""
    validate_schema_contract()

    should_stop_spark = False
    if spark is None:
        spark = get_spark("LoanDefaultPrediction-BronzeIngestion-All")
        should_stop_spark = True

    results: dict[str, Any] = {}
    total_start = time.time()
    table_names = list(TABLES.keys())
    total_tables = len(table_names)

    logger.info(f"Starting Bronze ingestion for {total_tables} tables...")

    try:
        for idx, tbl_name in enumerate(table_names, 1):
            print(f"[{idx}/{total_tables}] Ingesting {tbl_name} -> data/bronze/{tbl_name}/ ...", flush=True)
            res = ingest_table(tbl_name, spark=spark)
            results[tbl_name] = res
            print(
                f"       -> PASS: {res['bronze_rows']:,} rows | "
                f"{res['output_size_mb']} MB | {res['total_duration_seconds']:.2f}s",
                flush=True,
            )

        total_runtime = time.time() - total_start
        total_rows = sum(r["bronze_rows"] for r in results.values())
        total_bytes = sum(r["output_size_bytes"] for r in results.values())
        total_mb = round(total_bytes / (1024 * 1024), 2)

        summary: dict[str, Any] = {
            "tables_ingested": len(results),
            "total_tables": total_tables,
            "all_passed": all(r["status"] == "PASS" for r in results.values()),
            "total_rows": total_rows,
            "total_size_bytes": total_bytes,
            "total_size_mb": total_mb,
            "total_runtime_seconds": round(total_runtime, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tables": results,
        }

        # Write reports
        write_ingestion_reports(summary)
        return summary

    finally:
        if should_stop_spark:
            spark.stop()


def write_ingestion_reports(summary: dict[str, Any]) -> None:
    """Write ingestion_summary.json and summary.md reports to reports/bronze/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_DIR / "ingestion_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Written ingestion JSON summary: {json_path}")

    md_path = REPORTS_DIR / "summary.md"
    md_lines: list[str] = [
        "# Phase 2 Bronze Ingestion Summary",
        "",
        "> **Project:** Loan Default Prediction Using Big Data Analytics  ",
        f"> **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"> **Total Rows Ingested:** {summary['total_rows']:,}  ",
        f"> **Total Bronze Size:** {summary['total_size_mb']:,.2f} MB  ",
        f"> **Total Runtime:** {summary['total_runtime_seconds']:.2f} seconds  ",
        "",
        "---",
        "",
        "## 1. Bronze Objective & Scope",
        "",
        "The Bronze layer converts all 8 raw Home Credit CSV files into typed, validated Parquet files under `data/bronze/`. "
        "Strict Phase 1 `StructType` schemas are enforced without runtime schema inference (`inferSchema=False`). "
        "Raw data is preserved semantically: no cleaning, imputation, sentinel replacement, deduplication, or filtering is performed in Phase 2.",
        "",
        "---",
        "",
        "## 2. Ingestion & Validation Results",
        "",
        "| Table Name | Source Rows | Bronze Rows | Row Count Match | Schema Match | Files | Size (MB) | Runtime (s) | Status |",
        "| :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | :---: |",
    ]

    for tbl, r in summary["tables"].items():
        row_match = "PASS" if r["row_count_match"] else "FAIL"
        sch_match = "PASS" if r["schema_match"] else "FAIL"
        md_lines.append(
            f"| `{tbl}` | {r['source_rows']:,} | {r['bronze_rows']:,} | "
            f"{row_match} | {sch_match} | {r['output_file_count']} | "
            f"{r['output_size_mb']:.2f} | {r['total_duration_seconds']:.2f}s | **{r['status']}** |"
        )

    md_lines.extend([
        f"| **TOTAL** | **{summary['total_rows']:,}** | **{summary['total_rows']:,}** | **PASS** | **PASS** | **—** | **{summary['total_size_mb']:.2f} MB** | **{summary['total_runtime_seconds']:.2f}s** | **PASS** |",
        "",
        "---",
        "",
        "## 3. Partitioning & File Strategy",
        "",
        "Partition counts were configured to prevent thousands of small files while ensuring balanced parallelism on a 16 GB laptop:",
        "",
        "- `bureau_balance` (27.3M rows): 32 partitions",
        "- `installments_payments` (13.6M rows): 16 partitions",
        "- `POS_CASH_balance` (10.0M rows): 12 partitions",
        "- `credit_card_balance` (3.8M rows): 8 partitions",
        "- `bureau` (1.7M rows): 4 partitions",
        "- `previous_application` (1.67M rows): 4 partitions",
        "- `application_train` (307k rows): 2 partitions",
        "- `application_test` (48k rows): 1 partition",
        "",
        "---",
        "",
        "## 4. Idempotency & Rerunnability",
        "",
        "All Bronze tables are written with `.mode('overwrite')`. Rerunning `scripts/run_phase2_bronze.py` cleanly replaces existing Parquet partitions without accumulating stale or duplicated files.",
        "",
        "---",
        "",
        "## 5. Dataset Integrity",
        "",
        "All source CSV files in `dataset/` were accessed in strict read-only mode. No files were modified, renamed, moved, or deleted.",
        "",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Written markdown summary: {md_path}")

    # Also write ingestion_summary.md as specified in prompt
    alt_md_path = REPORTS_DIR / "ingestion_summary.md"
    with open(alt_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Written ingestion markdown summary: {alt_md_path}")
