"""Comprehensive Unit and Integration Tests for Phase 6 Splitting & Preprocessing.

Tests:
1. 70/15/15 split proportions approximately correct.
2. No split overlap across partitions.
3. Union of split IDs equals complete dataset.
4. TARGET exists in all supervised splits.
5. Stratification preserved across splits.
6. random_state=42 reproducibility across runs.
7. Different seed produces different split partitions.
8. Preprocessing fit only on TRAIN (train median != val median proof).
9. Validation data does not affect fitted imputer.
10. Test data does not affect fitted imputer.
11. OneHotEncoder handles unseen categories without error.
12. Feature lists are consistent and disjoint.
13. SK_ID_CURR excluded from model features.
14. TARGET excluded from model features.
15. Preprocessing artifacts can be reloaded and transform identically.
16. Integration: Split index files exist and have correct sizes (307,511 total).
17. Integration: Preprocessing reports and artifacts exist and are valid.
"""

import json
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_SPLITS_DIR = ROOT / "data" / "splits"
ARTIFACTS_DIR = ROOT / "artifacts" / "preprocessing"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"

from src.ml.preprocessing import (
    build_logistic_preprocessor,
    build_tree_preprocessing_metadata,
    fit_preprocessors,
    identify_feature_types,
    load_preprocessing_artifacts,
    save_preprocessing_artifacts,
)
from src.ml.split import (
    create_stratified_split,
    load_split_indices,
    save_split_indices,
    validate_split,
)


# ======================================================================
# Fixtures & Synthetic Datasets
# ======================================================================

@pytest.fixture
def synthetic_loan_data() -> pd.DataFrame:
    """Create a synthetic applicant dataset mimicking Home Credit distributions."""
    np.random.seed(42)
    n = 10_000
    ids = np.arange(100_000, 100_000 + n)
    # 8% default rate
    target = np.random.choice([0, 1], size=n, p=[0.92, 0.08])
    num1 = np.random.normal(50000, 15000, size=n)
    num2 = np.random.exponential(1000, size=n)
    cat1 = np.random.choice(["Cash", "Revolving"], size=n, p=[0.90, 0.10])
    cat2 = np.random.choice(["Secondary", "Higher", "Incomplete"], size=n, p=[0.70, 0.25, 0.05])

    df = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": target,
        "AMT_INCOME": num1,
        "AMT_CREDIT": num2,
        "NAME_CONTRACT_TYPE": cat1,
        "NAME_EDUCATION_TYPE": cat2,
    })
    return df


# ======================================================================
# Unit Tests: Splitting & Stratification
# ======================================================================

def test_split_proportions_approx(synthetic_loan_data: pd.DataFrame):
    """Verify 70/15/15 proportions are approximately respected."""
    train_df, val_df, test_df = create_stratified_split(synthetic_loan_data, random_state=42)
    total = len(synthetic_loan_data)

    train_pct = len(train_df) / total
    val_pct = len(val_df) / total
    test_pct = len(test_df) / total

    assert 0.69 <= train_pct <= 0.71, f"Train proportion {train_pct:.4f} out of bounds"
    assert 0.14 <= val_pct <= 0.16, f"Validation proportion {val_pct:.4f} out of bounds"
    assert 0.14 <= test_pct <= 0.16, f"Test proportion {test_pct:.4f} out of bounds"


def test_no_split_overlap(synthetic_loan_data: pd.DataFrame):
    """Verify all partitions are mutually exclusive (zero overlap)."""
    train_df, val_df, test_df = create_stratified_split(synthetic_loan_data, random_state=42)

    train_ids = set(train_df["SK_ID_CURR"])
    val_ids = set(val_df["SK_ID_CURR"])
    test_ids = set(test_df["SK_ID_CURR"])

    assert len(train_ids.intersection(val_ids)) == 0, "Overlap between train and validation!"
    assert len(train_ids.intersection(test_ids)) == 0, "Overlap between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Overlap between validation and test!"


def test_union_equals_complete(synthetic_loan_data: pd.DataFrame):
    """Verify complete applicant population conservation (no loss, no duplication)."""
    train_df, val_df, test_df = create_stratified_split(synthetic_loan_data, random_state=42)
    all_orig_ids = set(synthetic_loan_data["SK_ID_CURR"])
    union_ids = set(train_df["SK_ID_CURR"]) | set(val_df["SK_ID_CURR"]) | set(test_df["SK_ID_CURR"])

    assert union_ids == all_orig_ids, "Split union does not match original IDs!"
    assert len(train_df) + len(val_df) + len(test_df) == len(synthetic_loan_data)


def test_target_exists_in_all_supervised_splits(synthetic_loan_data: pd.DataFrame):
    """Verify TARGET exists and has both positive and negative classes in all partitions."""
    train_df, val_df, test_df = create_stratified_split(synthetic_loan_data, random_state=42)

    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert "TARGET" in sdf.columns, f"TARGET missing in {name}"
        classes = set(sdf["TARGET"].unique())
        assert {0, 1}.issubset(classes), f"{name} missing binary classes: {classes}"


def test_stratification_preserved(synthetic_loan_data: pd.DataFrame):
    """Verify target default rate is preserved across all three splits."""
    train_df, val_df, test_df = create_stratified_split(synthetic_loan_data, random_state=42)
    total_pos_rate = synthetic_loan_data["TARGET"].mean()

    train_pos_rate = train_df["TARGET"].mean()
    val_pos_rate = val_df["TARGET"].mean()
    test_pos_rate = test_df["TARGET"].mean()

    assert abs(train_pos_rate - total_pos_rate) < 0.005, f"Train rate delta too large: {train_pos_rate} vs {total_pos_rate}"
    assert abs(val_pos_rate - total_pos_rate) < 0.005, f"Val rate delta too large: {val_pos_rate} vs {total_pos_rate}"
    assert abs(test_pos_rate - total_pos_rate) < 0.005, f"Test rate delta too large: {test_pos_rate} vs {total_pos_rate}"


def test_random_state_reproducibility(synthetic_loan_data: pd.DataFrame):
    """Verify running split twice with random_state=42 produces identical memberships."""
    train_1, val_1, test_1 = create_stratified_split(synthetic_loan_data, random_state=42)
    train_2, val_2, test_2 = create_stratified_split(synthetic_loan_data, random_state=42)

    assert list(train_1["SK_ID_CURR"]) == list(train_2["SK_ID_CURR"])
    assert list(val_1["SK_ID_CURR"]) == list(val_2["SK_ID_CURR"])
    assert list(test_1["SK_ID_CURR"]) == list(test_2["SK_ID_CURR"])


def test_different_seed_produces_different_split(synthetic_loan_data: pd.DataFrame):
    """Verify a different seed produces different split partitions."""
    train_1, val_1, _ = create_stratified_split(synthetic_loan_data, random_state=42)
    train_2, val_2, _ = create_stratified_split(synthetic_loan_data, random_state=999)

    assert set(train_1["SK_ID_CURR"]) != set(train_2["SK_ID_CURR"])
    assert set(val_1["SK_ID_CURR"]) != set(val_2["SK_ID_CURR"])


# ======================================================================
# Unit Tests: Preprocessing & Leakage Prevention
# ======================================================================

def test_preprocessing_fit_only_on_train():
    """Explicitly prove that preprocessing is fitted on TRAIN only.

    Synthetic test:
    - TRAIN numeric feature has values with median = 10.0
    - VALIDATION numeric feature has values with median = 500.0
    - Transform VALIDATION containing a NaN value
    - Verify imputed value in transformed validation is 10.0 (TRAIN median), NOT 500.0 (VAL median).
    """
    train_data = pd.DataFrame({
        "num": [8.0, 10.0, 12.0],  # median = 10.0, mean = 10.0, std = 1.63299
        "cat": ["A", "A", "B"],
    })
    val_data = pd.DataFrame({
        "num": [400.0, 500.0, 600.0, np.nan],  # val median = 500.0
        "cat": ["A", "B", "A", "A"],
    })

    preprocessor = build_logistic_preprocessor(["num"], ["cat"])
    # Fit ONLY on TRAIN
    preprocessor.fit(train_data)

    # Imputer from pipeline
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[0] == 10.0, f"Expected train median 10.0, got {imputer.statistics_[0]}"

    # Transform validation data
    transformed_val = preprocessor.transform(val_data)

    # The 4th row had NaN in 'num'. Since it was imputed with train median (10.0),
    # and standard-scaled with train (mean=10.0, std=1.63299), (10.0 - 10.0)/std = 0.0!
    # If it had been imputed with val median (500.0), the standardized value would be (500-10)/std ~= 300!
    imputed_scaled_val = transformed_val[3, 0]
    assert np.isclose(imputed_scaled_val, 0.0, atol=1e-5), (
        f"Imputed scaled value was {imputed_scaled_val}, expected 0.0 (train median standardization)"
    )


def test_validation_does_not_affect_fitted_imputer():
    """Verify transforming validation data does not alter fitted statistics."""
    train_data = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["A", "A", "B"]})
    val_wild = pd.DataFrame({"num": [1e9, 1e9, 1e9], "cat": ["B", "B", "B"]})

    preprocessor = build_logistic_preprocessor(["num"], ["cat"])
    preprocessor.fit(train_data)

    initial_median = preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_[0]
    initial_mean = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_[0]

    # Transform wild validation data
    _ = preprocessor.transform(val_wild)

    # Verify statistics remain identical
    assert preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_[0] == initial_median
    assert preprocessor.named_transformers_["num"].named_steps["scaler"].mean_[0] == initial_mean


def test_test_does_not_affect_fitted_imputer():
    """Verify transforming test data does not alter fitted statistics."""
    train_data = pd.DataFrame({"num": [10.0, 20.0, 30.0], "cat": ["A", "B", "C"]})
    test_wild = pd.DataFrame({"num": [-9999.0, 9999.0], "cat": ["A", "B"]})

    preprocessor = build_logistic_preprocessor(["num"], ["cat"])
    preprocessor.fit(train_data)

    initial_median = preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_[0]
    _ = preprocessor.transform(test_wild)
    assert preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_[0] == initial_median


def test_onehot_encoder_handles_unseen_categories():
    """Verify OneHotEncoder with handle_unknown='ignore' handles novel categories smoothly."""
    train_data = pd.DataFrame({"num": [1.0, 2.0], "cat": ["KnownA", "KnownB"]})
    val_data = pd.DataFrame({"num": [3.0, 4.0], "cat": ["UnknownX", "UnknownY"]})

    preprocessor = build_logistic_preprocessor(["num"], ["cat"])
    preprocessor.fit(train_data)

    # Should not raise exception
    transformed_val = preprocessor.transform(val_data)
    assert transformed_val.shape[0] == 2
    # Categorical columns in transformed_val should be all zeros for unseen categories
    assert np.all(transformed_val[:, 1:] == 0.0)


def test_feature_lists_are_consistent():
    """Verify feature classification is complete, non-overlapping, and excludes ID and Label."""
    registry_path = CONFIGS_DIR / "features.yaml"
    if not registry_path.exists():
        pytest.skip("configs/features.yaml not present")

    res = identify_feature_types(registry_path)

    # Excluded check
    assert "SK_ID_CURR" in res["excluded_features"]
    assert "TARGET" in res["excluded_features"]
    assert "SK_ID_CURR" not in res["all_model_features"]
    assert "TARGET" not in res["all_model_features"]

    # Disjointness
    num_set = set(res["numeric_features"])
    cat_set = set(res["categorical_features"])
    assert len(num_set.intersection(cat_set)) == 0

    # Total check
    assert res["numeric_count"] + res["categorical_count"] == len(res["all_model_features"])


def test_id_excluded_from_model_features():
    """Verify SK_ID_CURR is excluded from model feature sets."""
    registry_path = CONFIGS_DIR / "features.yaml"
    if not registry_path.exists():
        pytest.skip("configs/features.yaml not present")

    res = identify_feature_types(registry_path)
    assert "SK_ID_CURR" not in res["numeric_features"]
    assert "SK_ID_CURR" not in res["categorical_features"]
    assert "SK_ID_CURR" not in res["all_model_features"]


def test_target_excluded_from_model_features():
    """Verify TARGET is excluded from model feature sets."""
    registry_path = CONFIGS_DIR / "features.yaml"
    if not registry_path.exists():
        pytest.skip("configs/features.yaml not present")

    res = identify_feature_types(registry_path)
    assert "TARGET" not in res["numeric_features"]
    assert "TARGET" not in res["categorical_features"]
    assert "TARGET" not in res["all_model_features"]


def test_preprocessing_artifacts_reloadable():
    """Verify saved ColumnTransformer and metadata can be reloaded and transform identically."""
    train_data = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["X", "Y", "X"]})
    test_data = pd.DataFrame({"num": [2.0, np.nan], "cat": ["Y", "X"]})

    fitted_ct, meta = fit_preprocessors(train_data, ["num"], ["cat"])
    expected_transform = fitted_ct.transform(test_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_preprocessing_artifacts(
            fitted_ct, meta, ["num"], ["cat"], output_dir=pathlib.Path(tmpdir)
        )
        reloaded_ct, reloaded_meta = load_preprocessing_artifacts(pathlib.Path(tmpdir))

        reloaded_transform = reloaded_ct.transform(test_data)
        np.testing.assert_allclose(expected_transform, reloaded_transform)
        assert reloaded_meta["fitted_on"] == "TRAIN_ONLY"


# ======================================================================
# Integration Tests: Physical Split Datasets & Artifacts
# ======================================================================

def split_indices_exist() -> bool:
    """Check if Phase 6 split index files exist."""
    return (
        (DATA_SPLITS_DIR / "train_indices.parquet").exists()
        and (DATA_SPLITS_DIR / "val_indices.parquet").exists()
        and (DATA_SPLITS_DIR / "test_indices.parquet").exists()
    )


require_split_indices = pytest.mark.skipif(
    not split_indices_exist(),
    reason="Split indices not yet generated; run scripts/run_phase6_split.py first",
)


@require_split_indices
def test_split_indices_files_exist_and_valid():
    """Verify physical split indices parquet files exist and satisfy 307,511 applicant conservation."""
    train_idx, val_idx, test_idx = load_split_indices(DATA_SPLITS_DIR)

    train_cnt = len(train_idx)
    val_cnt = len(val_idx)
    test_cnt = len(test_idx)
    total_cnt = train_cnt + val_cnt + test_cnt

    assert total_cnt == 307_511, f"Expected 307,511 total applicants, got {total_cnt:,}"
    assert 215_000 <= train_cnt <= 216_000, f"Train count {train_cnt:,} unexpected"
    assert 46_000 <= val_cnt <= 47_000, f"Val count {val_cnt:,} unexpected"
    assert 46_000 <= test_cnt <= 47_000, f"Test count {test_cnt:,} unexpected"

    train_ids = set(train_idx["SK_ID_CURR"])
    val_ids = set(val_idx["SK_ID_CURR"])
    test_ids = set(test_idx["SK_ID_CURR"])

    assert len(train_ids & val_ids) == 0, "Train and Val overlap in persisted files!"
    assert len(train_ids & test_ids) == 0, "Train and Test overlap in persisted files!"
    assert len(val_ids & test_ids) == 0, "Val and Test overlap in persisted files!"


@require_split_indices
def test_split_and_preprocessing_reports_exist():
    """Verify JSON and Markdown reports are produced for split and preprocessing."""
    assert (REPORTS_DIR / "split_report.json").exists()
    assert (REPORTS_DIR / "split_report.md").exists()
    assert (REPORTS_DIR / "preprocessing_report.json").exists()
    assert (REPORTS_DIR / "preprocessing_report.md").exists()

    with open(REPORTS_DIR / "split_report.json", encoding="utf-8") as f:
        sdata = json.load(f)
    assert sdata["status"] == "PASS"
    assert sdata["total_applicants"] == 307_511

    with open(REPORTS_DIR / "preprocessing_report.json", encoding="utf-8") as f:
        pdata = json.load(f)
    assert pdata["status"] == "PASS"
    assert pdata["leakage_invariants"]["fitted_on"] == "TRAIN_ONLY"


@require_split_indices
def test_preprocessing_artifacts_exist_and_loadable():
    """Verify serialized preprocessing pipeline and text feature lists exist and load."""
    assert (ARTIFACTS_DIR / "fitted_logistic_preprocessor.joblib").exists()
    assert (ARTIFACTS_DIR / "preprocessing_metadata.json").exists()
    assert (ARTIFACTS_DIR / "numeric_features.txt").exists()
    assert (ARTIFACTS_DIR / "categorical_features.txt").exists()
    assert (ARTIFACTS_DIR / "transformed_feature_names.txt").exists()

    ct, meta = load_preprocessing_artifacts(ARTIFACTS_DIR)
    assert isinstance(ct, ColumnTransformer)
    assert meta["fitted_on"] == "TRAIN_ONLY"
    assert meta["train_samples"] > 200_000
