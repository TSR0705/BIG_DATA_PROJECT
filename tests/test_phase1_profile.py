"""Tests for Phase 1: Dataset Profiling & Schema Contract.

Verifies that all required artifacts, schemas, audits, and reports exist and are structurally valid
without re-loading the 58M raw rows.
"""

import json
import pathlib
import pytest
from pyspark.sql.types import StructType

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports" / "profiling"

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


def test_tables_registered():
    """Verify all 8 tables are defined in the central table registry."""
    from configs.tables import TABLES

    assert len(TABLES) == 8
    for tbl in EXPECTED_TABLES:
        assert tbl in TABLES
        assert pathlib.Path(TABLES[tbl]["path"]).exists(), f"File does not exist: {TABLES[tbl]['path']}"


def test_profiling_json_artifacts_exist():
    """Verify all 8 table profiling JSON files exist and have valid structure."""
    for tbl in EXPECTED_TABLES:
        json_file = REPORTS_DIR / f"{tbl}.json"
        assert json_file.exists(), f"Missing profiling JSON: {json_file}"
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["table"] == tbl
        assert data["rows"] > 0
        assert data["columns"] > 0
        assert len(data["columns_profile"]) == data["columns"]


def test_summary_markdown_exists():
    """Verify summary.md exists and contains key section headings."""
    summary_file = REPORTS_DIR / "summary.md"
    assert summary_file.exists()
    content = summary_file.read_text(encoding="utf-8")
    assert "# Phase 1 Dataset Profiling Summary" in content
    assert "## 2. Table Scale" in content
    assert "## 3. Primary Key Audit" in content
    assert "## 5. Foreign Key Coverage" in content
    assert "## 7. Anomaly Inventory" in content
    assert "## 10. Schema Contract" in content


def test_relationship_report_exists():
    """Verify relationships.json exists and contains 6 measured relationships."""
    rel_file = REPORTS_DIR / "relationships.json"
    assert rel_file.exists()
    with open(rel_file, encoding="utf-8") as f:
        rels = json.load(f)
    assert len(rels) == 6
    for r in rels:
        assert "key_coverage_pct" in r
        assert "row_coverage_pct" in r
        assert r["total_distinct_child_keys"] > 0


def test_anomaly_report_exists():
    """Verify anomalies.json exists and contains expected anomaly scans."""
    anom_file = REPORTS_DIR / "anomalies.json"
    assert anom_file.exists()
    with open(anom_file, encoding="utf-8") as f:
        anom = json.load(f)
    assert "days_employed_sentinel" in anom
    assert "xna_occurrences" in anom
    assert "days_columns_ranges" in anom
    assert "income_distribution" in anom
    assert anom["days_employed_sentinel"]["application_train"]["count"] > 0


def test_column_dictionary_exists():
    """Verify column_dictionary.json exists and has parsed entries."""
    dict_file = REPORTS_DIR / "column_dictionary.json"
    assert dict_file.exists()
    with open(dict_file, encoding="utf-8") as f:
        col_dict = json.load(f)
    assert len(col_dict) > 0


def test_schemas_py_imports_and_validates():
    """Verify configs/schemas.py imports and validate_schema_contract() passes."""
    from configs.schemas import TABLE_SCHEMAS, validate_schema_contract

    assert len(TABLE_SCHEMAS) == 8
    for tbl in EXPECTED_TABLES:
        assert tbl in TABLE_SCHEMAS
        assert isinstance(TABLE_SCHEMAS[tbl], StructType)

    assert validate_schema_contract() is True


def test_required_key_columns_in_schemas():
    """Verify primary and foreign key columns exist in schemas."""
    from configs.schemas import (
        APPLICATION_TEST_SCHEMA,
        APPLICATION_TRAIN_SCHEMA,
        BUREAU_BALANCE_SCHEMA,
        BUREAU_SCHEMA,
        PREVIOUS_APPLICATION_SCHEMA,
    )

    train_cols = [f.name for f in APPLICATION_TRAIN_SCHEMA.fields]
    test_cols = [f.name for f in APPLICATION_TEST_SCHEMA.fields]
    bureau_cols = [f.name for f in BUREAU_SCHEMA.fields]
    prev_cols = [f.name for f in PREVIOUS_APPLICATION_SCHEMA.fields]
    bb_cols = [f.name for f in BUREAU_BALANCE_SCHEMA.fields]

    assert "SK_ID_CURR" in train_cols
    assert "SK_ID_CURR" in test_cols
    assert "SK_ID_BUREAU" in bureau_cols
    assert "SK_ID_CURR" in bureau_cols
    assert "SK_ID_PREV" in prev_cols
    assert "SK_ID_CURR" in prev_cols
    assert "SK_ID_BUREAU" in bb_cols
    assert "MONTHS_BALANCE" in bb_cols


def test_no_duplicate_column_names_in_schemas():
    """Verify there are no duplicate field names in any table schema."""
    from configs.schemas import TABLE_SCHEMAS

    for name, schema in TABLE_SCHEMAS.items():
        names = [f.name for f in schema.fields]
        assert len(names) == len(set(names)), f"Duplicates in {name}: {len(names)} != {len(set(names))}"
