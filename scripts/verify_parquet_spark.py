import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.spark_session import load_spark_config, setup_spark_environment
from pyspark.sql import SparkSession

cfg = load_spark_config()
setup_spark_environment(cfg)

# Connect to the running SparkSession or create a local one
spark = (
    SparkSession.builder
    .appName("ParquetDataVerifier")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

layers = [
    ("data/bronze/application_train", "BRONZE application_train"),
    ("data/silver/application_train", "SILVER application_train"),
    ("data/gold/bureau", "GOLD bureau (aggregated)"),
    ("data/model_input/model_input.parquet", "MODEL INPUT (final 256-feature matrix)"),
    ("data/clustering/clustering_input.parquet", "CLUSTERING INPUT (unsupervised)"),
    ("data/gold/model_input_test", "GOLD model_input_test (inference spine)")
]

for path_str, label in layers:
    print("=" * 70)
    print(f"VERIFYING VIA SPARK: {label}")
    print(f"Path: {path_str}")
    print("=" * 70)
    p = ROOT / path_str
    if not p.exists():
        print(f"Directory {path_str} does not exist!")
        continue
    
    df = spark.read.parquet(str(p))
    row_count = df.count()
    col_count = len(df.columns)
    print(f"Row Count    : {row_count:,}")
    print(f"Column Count : {col_count}")
    print(f"Columns      : {df.columns[:6]} ... {df.columns[-3:]}")
    
    # Check TARGET presence
    has_target = "TARGET" in df.columns
    print(f"Has TARGET   : {has_target}")
    
    # Show sample
    sample_cols = [c for c in ["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT"] if c in df.columns]
    if not sample_cols:
        sample_cols = df.columns[:4]
    df.select(sample_cols).show(2, truncate=False)
    print()

spark.stop()
