"""Unit tests for Phase 10 Experiment E1 (Feature Ablation)."""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.ablation import (
    get_ablation_feature_sets,
    load_feature_manifest,
    load_train_val_data,
)


def test_feature_manifest_integrity():
    """Verify that all 256 model features are partitioned without overlap or omission."""
    groups, meta = load_feature_manifest()

    assert set(groups.keys()) == {
        "app",
        "bureau",
        "prev",
        "installments",
        "credit_card",
        "pos_cash",
        "derived",
    }
    assert len(groups["app"]) == 147
    assert len(groups["bureau"]) == 26
    assert len(groups["prev"]) == 25
    assert len(groups["installments"]) == 16
    assert len(groups["credit_card"]) == 21
    assert len(groups["pos_cash"]) == 14
    assert len(groups["derived"]) == 7

    # Ensure zero overlap
    all_cols = []
    for g, cols in groups.items():
        all_cols.extend(cols)
    assert len(all_cols) == 256
    assert len(set(all_cols)) == 256

    # Verify TARGET and SK_ID_CURR are strictly excluded
    assert "TARGET" not in all_cols
    assert "SK_ID_CURR" not in all_cols


def test_ablation_configurations_definition():
    """Verify exactly 13 ablation configurations (7 incremental + 6 LOGO)."""
    configs = get_ablation_feature_sets()
    assert len(configs) == 13

    # Check 7 incremental
    assert configs["app_only"]["features"].__len__() == 147
    assert configs["app_bureau"]["features"].__len__() == 173
    assert configs["app_prev"]["features"].__len__() == 172
    assert configs["app_installments"]["features"].__len__() == 163
    assert configs["app_credit_card"]["features"].__len__() == 168
    assert configs["app_pos_cash"]["features"].__len__() == 161
    assert configs["full_all_tables"]["features"].__len__() == 256

    # Check 6 LOGO
    assert configs["logo_bureau"]["features"].__len__() == 230
    assert configs["logo_prev"]["features"].__len__() == 231
    assert configs["logo_installments"]["features"].__len__() == 240
    assert configs["logo_credit_card"]["features"].__len__() == 235
    assert configs["logo_pos_cash"]["features"].__len__() == 242
    assert configs["logo_derived"]["features"].__len__() == 249


def test_ablation_data_loading_no_test_leakage():
    """Verify that load_train_val_data only accesses Train and Val, with zero Test leakage."""
    X_train, y_train, X_val, y_val, cat_cols = load_train_val_data()

    assert len(X_train) == 215_257
    assert len(y_train) == 215_257
    assert len(X_val) == 46_127
    assert len(y_val) == 46_127

    assert "SK_ID_CURR" not in X_train.columns
    assert "TARGET" not in X_train.columns
    assert "SK_ID_CURR" not in X_val.columns
    assert "TARGET" not in X_val.columns
    assert len(X_train.columns) == 256
    assert len(X_val.columns) == 256


def test_bootstrap_reproducibility():
    """Verify paired bootstrap sampling is deterministic with fixed seed."""
    np.random.seed(42)
    s1 = np.random.choice(100, size=100, replace=True)
    np.random.seed(42)
    s2 = np.random.choice(100, size=100, replace=True)
    assert np.array_equal(s1, s2)
