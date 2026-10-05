"""Tests for frozen Phase 1 / Phase 2 schemas and contracts."""

import pytest
from pyspark.sql.types import IntegerType, LongType, StringType, StructType

from configs.schemas import (
    APPLICATION_TEST_SCHEMA,
    APPLICATION_TRAIN_SCHEMA,
    BUREAU_BALANCE_SCHEMA,
    BUREAU_SCHEMA,
    CREDIT_CARD_BALANCE_SCHEMA,
    INSTALLMENTS_PAYMENTS_SCHEMA,
    POS_CASH_BALANCE_SCHEMA,
    PREVIOUS_APPLICATION_SCHEMA,
    TABLE_SCHEMAS,
    validate_schema_contract,
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


def test_all_eight_schemas_defined():
    """Verify that all 8 tables have defined StructType schemas."""
    assert len(TABLE_SCHEMAS) == 8
    for tbl in EXPECTED_TABLES:
        assert tbl in TABLE_SCHEMAS
        assert isinstance(TABLE_SCHEMAS[tbl], StructType)


def test_schema_contract_validation():
    """Verify validate_schema_contract() succeeds."""
    assert validate_schema_contract() is True


def test_no_duplicate_column_names():
    """Verify that no schema has duplicate column names."""
    for tbl, schema in TABLE_SCHEMAS.items():
        names = [f.name for f in schema.fields]
        assert len(names) == len(set(names)), f"Duplicate columns in {tbl}: {len(names)} != {len(set(names))}"


def test_expected_key_columns_exist():
    """Verify primary and foreign key columns are present with appropriate types."""
    # Application keys
    train_fields = {f.name: f.dataType for f in APPLICATION_TRAIN_SCHEMA.fields}
    test_fields = {f.name: f.dataType for f in APPLICATION_TEST_SCHEMA.fields}
    assert "SK_ID_CURR" in train_fields
    assert "TARGET" in train_fields
    assert "SK_ID_CURR" in test_fields

    # Bureau keys
    bureau_fields = {f.name: f.dataType for f in BUREAU_SCHEMA.fields}
    assert "SK_ID_BUREAU" in bureau_fields
    assert "SK_ID_CURR" in bureau_fields

    # Bureau balance keys
    bb_fields = {f.name: f.dataType for f in BUREAU_BALANCE_SCHEMA.fields}
    assert "SK_ID_BUREAU" in bb_fields
    assert "MONTHS_BALANCE" in bb_fields

    # Previous application keys
    prev_fields = {f.name: f.dataType for f in PREVIOUS_APPLICATION_SCHEMA.fields}
    assert "SK_ID_PREV" in prev_fields
    assert "SK_ID_CURR" in prev_fields

    # Monthly / history table keys
    inst_fields = {f.name: f.dataType for f in INSTALLMENTS_PAYMENTS_SCHEMA.fields}
    assert "SK_ID_PREV" in inst_fields

    cc_fields = {f.name: f.dataType for f in CREDIT_CARD_BALANCE_SCHEMA.fields}
    assert "SK_ID_PREV" in cc_fields
    assert "MONTHS_BALANCE" in cc_fields

    pos_fields = {f.name: f.dataType for f in POS_CASH_BALANCE_SCHEMA.fields}
    assert "SK_ID_PREV" in pos_fields
    assert "MONTHS_BALANCE" in pos_fields


def test_schema_field_counts():
    """Verify expected field counts across schemas."""
    assert len(APPLICATION_TRAIN_SCHEMA.fields) == 122
    assert len(APPLICATION_TEST_SCHEMA.fields) == 121
    assert len(BUREAU_SCHEMA.fields) == 17
    assert len(BUREAU_BALANCE_SCHEMA.fields) == 3
    assert len(PREVIOUS_APPLICATION_SCHEMA.fields) == 37
    assert len(INSTALLMENTS_PAYMENTS_SCHEMA.fields) == 8
    assert len(CREDIT_CARD_BALANCE_SCHEMA.fields) == 23
    assert len(POS_CASH_BALANCE_SCHEMA.fields) == 8
