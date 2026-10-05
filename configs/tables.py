"""Table registry defining the single source of truth for the 8 Home Credit tables."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"

TABLES = {
    "application_train": {
        "filename": "application_train.csv",
        "path": str(DATASET_DIR / "application_train.csv"),
        "primary_key": ["SK_ID_CURR"],
        "candidate_grain": ["SK_ID_CURR"],
        "grain": "one row per current application",
        "parent_relationship": None,
    },
    "application_test": {
        "filename": "application_test.csv",
        "path": str(DATASET_DIR / "application_test.csv"),
        "primary_key": ["SK_ID_CURR"],
        "candidate_grain": ["SK_ID_CURR"],
        "grain": "one row per current application",
        "parent_relationship": None,
    },
    "bureau": {
        "filename": "bureau.csv",
        "path": str(DATASET_DIR / "bureau.csv"),
        "primary_key": ["SK_ID_BUREAU"],
        "candidate_grain": ["SK_ID_BUREAU"],
        "grain": "one bureau record per bureau credit",
        "parent_relationship": {
            "parent_table": "application_train",
            "parent_key": "SK_ID_CURR",
            "child_key": "SK_ID_CURR",
        },
    },
    "bureau_balance": {
        "filename": "bureau_balance.csv",
        "path": str(DATASET_DIR / "bureau_balance.csv"),
        "primary_key": None,
        "candidate_grain": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
        "grain": "monthly record per SK_ID_BUREAU",
        "parent_relationship": {
            "parent_table": "bureau",
            "parent_key": "SK_ID_BUREAU",
            "child_key": "SK_ID_BUREAU",
        },
    },
    "previous_application": {
        "filename": "previous_application.csv",
        "path": str(DATASET_DIR / "previous_application.csv"),
        "primary_key": ["SK_ID_PREV"],
        "candidate_grain": ["SK_ID_PREV"],
        "grain": "one previous application",
        "parent_relationship": {
            "parent_table": "application_train",
            "parent_key": "SK_ID_CURR",
            "child_key": "SK_ID_CURR",
        },
    },
    "installments_payments": {
        "filename": "installments_payments.csv",
        "path": str(DATASET_DIR / "installments_payments.csv"),
        "primary_key": None,
        "candidate_grain": ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"],
        "grain": "installment/payment history",
        "parent_relationship": {
            "parent_table": "previous_application",
            "parent_key": "SK_ID_PREV",
            "child_key": "SK_ID_PREV",
        },
    },
    "credit_card_balance": {
        "filename": "credit_card_balance.csv",
        "path": str(DATASET_DIR / "credit_card_balance.csv"),
        "primary_key": None,
        "candidate_grain": ["SK_ID_PREV", "MONTHS_BALANCE"],
        "grain": "monthly credit-card history",
        "parent_relationship": {
            "parent_table": "previous_application",
            "parent_key": "SK_ID_PREV",
            "child_key": "SK_ID_PREV",
        },
    },
    "POS_CASH_balance": {
        "filename": "POS_CASH_balance.csv",
        "path": str(DATASET_DIR / "POS_CASH_balance.csv"),
        "primary_key": None,
        "candidate_grain": ["SK_ID_PREV", "MONTHS_BALANCE"],
        "grain": "monthly POS/CASH history",
        "parent_relationship": {
            "parent_table": "previous_application",
            "parent_key": "SK_ID_PREV",
            "child_key": "SK_ID_PREV",
        },
    },
}
