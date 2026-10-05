# data/ — generated data layers (not committed)

Everything in this folder is **regenerated from `dataset/` + code** and is git-ignored.

| Layer | Path | Produced by | Content |
|---|---|---|---|
| Bronze | `data/bronze/<table>/` | Phase 2 `src/ingestion/bronze.py` | Schema-enforced Parquet mirrors of the 8 raw CSVs |
| Silver | `data/silver/<table>/` | Phase 3 `src/preprocessing/silver.py` | Cleaned, typed, de-duplicated tables + data-quality report |
| Gold | `data/gold/features_base.parquet` | Phase 4 `src/features/assemble_gold.py` | One row per `SK_ID_CURR` (307,511), aggregate-before-join |
| Gold | `data/gold/model_input.parquet` | Phase 5 `src/features/` | Engineered features + `TARGET`, leakage-audited |
| Gold | `data/gold/split_indices.parquet` | Phase 6 `src/training/split.py` | Stratified 70/15/15 split indices (seed 42) |

`data/_verify.parquet/` is a throwaway written by `scripts/verify_env.py` (Phase 0 check).

The raw Kaggle files stay in `dataset/` (also git-ignored; ~3.4 GB) and are never modified.
