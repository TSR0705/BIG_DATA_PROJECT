"""Phase 6 Stratified Split & Model Preprocessing Runner.

Executes the official supervised-learning split and model-specific preprocessing layer:
1. Verifies input: data/model_input/model_input.parquet
2. Verifies raw dataset/ remains untouched
3. Identifies numeric and categorical features from configs/features.yaml
4. Creates stratified 70/15/15 split on TARGET (random_state=42)
5. Validates split invariants (disjointness, completeness, target preservation)
6. Persists split indices to data/splits/
7. Fits Logistic Regression preprocessor STRICTLY on X_train
8. Compiles tree models preprocessing metadata
9. Persists preprocessing artifacts to artifacts/preprocessing/
10. Generates split and preprocessing reports (JSON and Markdown)
11. Re-verifies raw dataset immutability
"""

import json
import os
import pathlib
import sys
import time
from typing import Any

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger
from src.ml.preprocessing import (
    ARTIFACTS_DIR,
    build_tree_preprocessing_metadata,
    fit_preprocessors,
    identify_feature_types,
    save_preprocessing_artifacts,
    write_preprocessing_reports,
)
from src.ml.split import (
    DATA_SPLITS_DIR,
    TOTAL_APPLICANTS,
    create_stratified_split,
    save_split_indices,
    validate_split,
    write_split_reports,
)

logger = get_logger("RunPhase6Split", "run_phase6_split.log")

DATASET_DIR = ROOT / "dataset"
MODEL_INPUT_DIR = ROOT / "data" / "model_input"
MODEL_INPUT_PATH = (
    (MODEL_INPUT_DIR / "model_input.parquet")
    if (MODEL_INPUT_DIR / "model_input.parquet").exists()
    else MODEL_INPUT_DIR
)
CONFIGS_DIR = ROOT / "configs"
REPORTS_DIR = ROOT / "reports"


def check_raw_dataset_timestamps() -> dict[str, float]:
    """Capture modification timestamps for all raw CSV files in dataset/."""
    mtimes = {}
    if DATASET_DIR.exists():
        for csv_file in DATASET_DIR.glob("*.csv"):
            mtimes[csv_file.name] = csv_file.stat().st_mtime
    return mtimes


def verify_input_dataset() -> None:
    """Verify that Phase 5 model input exists and contains Parquet data."""
    if not MODEL_INPUT_PATH.exists():
        raise FileNotFoundError(f"Model input directory missing: {MODEL_INPUT_PATH}")
    parquet_files = list(MODEL_INPUT_PATH.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in: {MODEL_INPUT_PATH}")


def main() -> int:
    start_time = time.time()
    logger.info("============================================================")
    logger.info("Starting Phase 6: Train/Val/Test Split & Model Preprocessing")
    logger.info("============================================================")

    # Step 1: Pre-flight checks
    try:
        logger.info("Verifying Phase 5 model input availability...")
        verify_input_dataset()
        logger.info(f"Model input verified present at {MODEL_INPUT_PATH}")
    except Exception as exc:
        logger.error(f"Input verification failed: {exc}", exc_info=True)
        print(f"ERROR: Input verification failed: {exc}", file=sys.stderr)
        return 1

    pre_mtimes = check_raw_dataset_timestamps()

    try:
        # Step 2: Load Model Input Dataset
        print("\n[1/6] Loading Phase 5 model input dataset ...", flush=True)
        t_load = time.time()
        dataset = pq.ParquetDataset(str(MODEL_INPUT_PATH))
        table = dataset.read()
        df = table.to_pandas()
        load_duration = time.time() - t_load

        total_rows, total_cols = df.shape
        logger.info(f"Loaded {total_rows:,} rows × {total_cols} columns in {load_duration:.2f}s")
        print(f"       Loaded {total_rows:,} rows × {total_cols} columns in {load_duration:.2f}s")

        if total_rows != TOTAL_APPLICANTS:
            raise AssertionError(
                f"Expected exactly {TOTAL_APPLICANTS:,} applicants, got {total_rows:,}"
            )

        # Step 3: Identify Feature Types
        print("[2/6] Identifying feature types from registry ...", flush=True)
        registry_path = CONFIGS_DIR / "features.yaml"
        feature_types = identify_feature_types(registry_path, columns=list(df.columns), df=df)
        print(
            f"       Features: {feature_types['numeric_count']} Numeric, "
            f"{feature_types['categorical_count']} Categorical, "
            f"{feature_types['excluded_count']} Excluded (ID & Label)"
        )

        # Step 4: Create Stratified Split
        print("[3/6] Generating stratified 70/15/15 split on TARGET (seed=42) ...", flush=True)
        train_df, val_df, test_df = create_stratified_split(
            df=df,
            target_col="TARGET",
            id_col="SK_ID_CURR",
            random_state=42,
            train_size=0.70,
            val_size=0.15,
            test_size=0.15,
        )

        # Step 5: Validate Split Invariants
        print("[4/6] Validating split invariants (disjointness & stratification) ...", flush=True)
        val_res = validate_split(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            id_col="SK_ID_CURR",
            target_col="TARGET",
            total_expected_rows=TOTAL_APPLICANTS,
        )
        print(
            f"       Train: {val_res['train_rows']:,} rows ({val_res['train_ratio']:.2%}) | "
            f"Default Rate: {val_res['stratification']['train']['positive_rate']:.3%}"
        )
        print(
            f"       Val  : {val_res['val_rows']:,} rows ({val_res['val_ratio']:.2%}) | "
            f"Default Rate: {val_res['stratification']['validation']['positive_rate']:.3%}"
        )
        print(
            f"       Test : {val_res['test_rows']:,} rows ({val_res['test_ratio']:.2%}) | "
            f"Default Rate: {val_res['stratification']['test']['positive_rate']:.3%}"
        )

        # Step 6: Persist Split Indices
        print("[5/6] Persisting split indices to data/splits/ ...", flush=True)
        save_split_indices(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=DATA_SPLITS_DIR,
            id_col="SK_ID_CURR",
        )

        # Step 7: Fit Preprocessors strictly on TRAIN
        print("[6/6] Fitting Logistic Regression preprocessor on TRAIN ONLY ...", flush=True)
        X_train = train_df[feature_types["all_model_features"]]
        fitted_preprocessor, preprocessor_meta = fit_preprocessors(
            X_train=X_train,
            numeric_features=feature_types["numeric_features"],
            categorical_features=feature_types["categorical_features"],
        )

        # Tree models metadata
        tree_meta = build_tree_preprocessing_metadata(
            numeric_features=feature_types["numeric_features"],
            categorical_features=feature_types["categorical_features"],
        )

        # Persist Preprocessing Artifacts
        save_preprocessing_artifacts(
            preprocessor=fitted_preprocessor,
            metadata=preprocessor_meta,
            numeric_features=feature_types["numeric_features"],
            categorical_features=feature_types["categorical_features"],
            output_dir=ARTIFACTS_DIR,
        )

        # Generate Reports
        write_split_reports(
            validation_results=val_res,
            random_state=42,
            source_dataset=str(MODEL_INPUT_PATH),
            output_dir=REPORTS_DIR,
        )

        write_preprocessing_reports(
            preprocessor_metadata=preprocessor_meta,
            feature_types=feature_types,
            tree_metadata=tree_meta,
            output_dir=REPORTS_DIR,
        )

        # Step 8: Verify raw dataset was untouched
        post_mtimes = check_raw_dataset_timestamps()
        for fname, pre_mtime in pre_mtimes.items():
            post_mtime = post_mtimes.get(fname)
            if post_mtime != pre_mtime:
                raise RuntimeError(
                    f"CRITICAL: Raw dataset file '{fname}' was modified during Phase 6! "
                    f"Pre: {pre_mtime}, Post: {post_mtime}"
                )

        total_runtime = time.time() - start_time
        print("\n" + "=" * 65)
        print("PHASE 6 SPLIT & PREPROCESSING PIPELINE COMPLETE")
        print(f"Total Applicants        : {TOTAL_APPLICANTS:,}")
        print(f"Train Partition         : {val_res['train_rows']:,} rows ({val_res['train_ratio']:.2%})")
        print(f"Validation Partition    : {val_res['val_rows']:,} rows ({val_res['val_ratio']:.2%})")
        print(f"Test Partition          : {val_res['test_rows']:,} rows ({val_res['test_ratio']:.2%})")
        print(f"Default Rates           : Train={val_res['stratification']['train']['positive_rate']:.3%}, "
              f"Val={val_res['stratification']['validation']['positive_rate']:.3%}, "
              f"Test={val_res['stratification']['test']['positive_rate']:.3%}")
        print(f"Model-Eligible Features : {len(feature_types['all_model_features'])} ({feature_types['numeric_count']} Num, {feature_types['categorical_count']} Cat)")
        print(f"Transformed Features    : {preprocessor_meta['output_features']['transformed_total_count']} (Logistic Pipeline)")
        print(f"Leakage Policy          : Preprocessors fitted STRICTLY ON TRAIN")
        print(f"Split Invariants        : 0 Overlap, 0 Row Loss, 100% Stratified")
        print(f"Raw Dataset Untouched   : Verified (0 modifications)")
        print(f"Pipeline Runtime        : {total_runtime:.2f}s")
        print(f"Status                  : PASS (GREEN)")
        print("=" * 65 + "\n")

        return 0

    except Exception as exc:
        logger.error(f"Phase 6 execution failed: {exc}", exc_info=True)
        print(f"\nERROR: Phase 6 execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
