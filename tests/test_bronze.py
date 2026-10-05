"""Tests for Phase 2 Bronze Parquet ingestion."""

import json
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BRONZE_DIR = ROOT / "data" / "bronze"
REPORTS_DIR = ROOT / "reports" / "bronze"
DATASET_DIR = ROOT / "dataset"

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

PHASE1_EXPECTED_COUNTS = {
    "application_train": 307_511,
    "application_test": 48_744,
    "bureau": 1_716_428,
    "bureau_balance": 27_299_925,
    "previous_application": 1_670_214,
    "installments_payments": 13_605_401,
    "credit_card_balance": 3_840_312,
    "POS_CASH_balance": 10_001_358,
}


def test_bronze_directories_exist():
    """Verify all 8 Bronze directories exist under data/bronze/."""
    assert BRONZE_DIR.exists(), f"Bronze base directory missing: {BRONZE_DIR}"
    for tbl in EXPECTED_TABLES:
        tbl_dir = BRONZE_DIR / tbl
        assert tbl_dir.exists(), f"Missing Bronze directory for {tbl}: {tbl_dir}"
        parquet_files = list(tbl_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, f"No Parquet part files found in {tbl_dir}"


def test_ingestion_summary_json_valid():
    """Verify reports/bronze/ingestion_summary.json exists and all tables passed."""
    summary_file = REPORTS_DIR / "ingestion_summary.json"
    assert summary_file.exists(), f"Missing metadata report: {summary_file}"

    with open(summary_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("tables_ingested") == 8
    assert data.get("all_passed") is True
    assert data.get("total_rows") == sum(PHASE1_EXPECTED_COUNTS.values())
    assert data.get("total_size_mb", 0) > 0

    tables = data.get("tables", {})
    assert len(tables) == 8
    for tbl, info in tables.items():
        assert info["status"] == "PASS"
        assert info["row_count_match"] is True
        assert info["schema_match"] is True
        assert info["bronze_rows"] == PHASE1_EXPECTED_COUNTS[tbl]
        assert info["output_file_count"] > 0
        assert info["output_size_bytes"] > 0


def test_ingestion_summary_markdown_valid():
    """Verify reports/bronze/summary.md and ingestion_summary.md exist."""
    summary_md = REPORTS_DIR / "summary.md"
    assert summary_md.exists()
    content = summary_md.read_text(encoding="utf-8")
    assert "# Phase 2 Bronze Ingestion Summary" in content
    assert "## 2. Ingestion & Validation Results" in content
    assert "## 4. Idempotency & Rerunnability" in content
    assert f"{sum(PHASE1_EXPECTED_COUNTS.values()):,}" in content


def test_bronze_parquet_readable_via_spark():
    """Verify that Bronze Parquet can be loaded and read back cleanly using Spark."""
    from src.common.spark_session import get_spark
    from configs.schemas import APPLICATION_TEST_SCHEMA

    spark = get_spark("TestBronzeReadable")
    try:
        test_path = str(BRONZE_DIR / "application_test")
        df = spark.read.parquet(test_path)
        assert df.count() == 48_744
        assert len(df.columns) == 121
        assert df.schema == APPLICATION_TEST_SCHEMA
    finally:
        spark.stop()


def test_raw_dataset_untouched():
    """Verify that dataset/ CSV files still exist with expected original sizes."""
    assert DATASET_DIR.exists()
    expected_csvs = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "previous_application.csv",
        "installments_payments.csv",
        "credit_card_balance.csv",
        "POS_CASH_balance.csv",
    ]
    for csv_file in expected_csvs:
        p = DATASET_DIR / csv_file
        assert p.exists(), f"Source CSV missing: {p}"
        assert p.stat().st_size > 10_000, f"Source CSV corrupted: {p}"
