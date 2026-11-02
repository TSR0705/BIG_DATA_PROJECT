import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

def inspect_parquet(path_str, name):
    print("=" * 70)
    print(f"INSPECTING: {name} ({path_str})")
    print("=" * 70)
    p = Path(path_str)
    if not p.exists():
        print(f"ERROR: {p} does not exist!")
        return
    
    # Read table / dataset
    try:
        ds = pq.ParquetDataset(str(p))
        cols = ds.schema.names
        print(f"Total Columns : {len(cols)}")
        print(f"Sample Columns: {cols[:8]} ... {cols[-4:]}")
        
        # Read a small sample of 3 rows
        table = pq.read_table(str(p))
        df_sample = table.slice(0, 3).to_pandas()
        print(f"Total Rows    : {len(table):,}")
        print(f"Sample Schema Data Types (first 5):")
        for col in cols[:5]:
            print(f"  - {col}: {table.schema.field(col).type}")
            
        print("\nSample Non-Sensitive Data (3 rows):")
        display_cols = [c for c in ["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT"] if c in cols]
        if not display_cols:
            display_cols = cols[:5]
        print(df_sample[display_cols].to_string())
        
        # Check TARGET presence
        has_target = "TARGET" in cols
        print(f"\nTARGET column present: {has_target}")
        if has_target:
            target_vals = df_sample["TARGET"].tolist()
            print(f"Sample TARGET values : {target_vals}")
    except Exception as e:
        print(f"Error inspecting {name}: {e}")

inspect_parquet("data/bronze/application_train", "BRONZE application_train")
inspect_parquet("data/silver/application_train", "SILVER application_train")
inspect_parquet("data/gold/bureau", "GOLD bureau (aggregated)")
inspect_parquet("data/model_input/model_input.parquet", "MODEL INPUT (final 256-feature matrix)")
inspect_parquet("data/clustering/clustering_input.parquet", "CLUSTERING INPUT (unsupervised)")
