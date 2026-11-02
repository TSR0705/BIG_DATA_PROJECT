"""Live Spark UI Demonstration script for Big Data Project.

Initializes a SparkSession with Spark UI enabled on port 4040+, executes genuine
DataFrame transformations and actions on existing project Parquet data, prints
the physical execution plan and exact Spark UI URL, and keeps the UI active
for interactive reviewer inspection.
"""

import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.spark_session import load_spark_config, setup_spark_environment
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    cfg = load_spark_config()
    setup_spark_environment(cfg)

    print("=" * 70)
    print("STARTING LIVE APACHE SPARK DEMONSTRATION ENGINE")
    print("=" * 70)
    print(f"PySpark Version : {pyspark.__version__}")
    print(f"Python Runtime  : {sys.executable}")
    print(f"JAVA_HOME       : {os.environ.get('JAVA_HOME')}")
    print(f"HADOOP_HOME     : {os.environ.get('HADOOP_HOME')}")

    # Build demonstration session with Spark UI explicitly enabled
    spark = (
        SparkSession.builder
        .appName("LoanDefaultPrediction-LiveReviewerDemo")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    ui_url = sc.uiWebUrl
    print(f"\n[SPARK UI ACTIVE] Real Web UI is listening at:")
    print(f"===> {ui_url} <===")
    print("Open this URL in your web browser to view Jobs, Stages, Executors & Environment.\n")

    # ------------------------------------------------------------------
    # DEMO JOB 1: Materialized Bronze Parquet Read & Action
    # ------------------------------------------------------------------
    print("-" * 70)
    print("TRIGGERING SPARK JOB 1: Materialized Bronze Parquet Read & Count")
    print("-" * 70)
    bronze_train_path = ROOT / "data" / "bronze" / "application_train"
    print(f"Reading: {bronze_train_path}")
    df_bronze = spark.read.parquet(str(bronze_train_path))
    print(f"Schema columns count: {len(df_bronze.columns)}")
    
    t0 = time.time()
    train_count = df_bronze.count()  # Spark Action
    t_count = time.time() - t0
    print(f"Job 1 Action: df_bronze.count() = {train_count:,} rows (completed in {t_count:.2f}s)")

    # ------------------------------------------------------------------
    # DEMO JOB 2: GroupBy Aggregation & Shuffle
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TRIGGERING SPARK JOB 2: GroupBy Aggregation & Shuffle (Contract Types)")
    print("-" * 70)
    agg_df = (
        df_bronze.groupBy("NAME_CONTRACT_TYPE")
        .agg(
            F.count("*").alias("APPLICANT_COUNT"),
            F.round(F.avg("AMT_CREDIT"), 2).alias("AVG_CREDIT"),
            F.round(F.avg("AMT_INCOME_TOTAL"), 2).alias("AVG_INCOME"),
            F.round(F.min("AMT_CREDIT"), 2).alias("MIN_CREDIT"),
            F.round(F.max("AMT_CREDIT"), 2).alias("MAX_CREDIT"),
        )
    )
    print("Aggregated DataFrame Action (.show()):")
    agg_df.show(truncate=False)

    # ------------------------------------------------------------------
    # DEMO JOB 3: Relational Multi-Table Join & Physical Catalyst Plan
    # ------------------------------------------------------------------
    print("-" * 70)
    print("TRIGGERING SPARK JOB 3: Multi-Table Relational Join (Application + Gold Bureau)")
    print("-" * 70)
    gold_bureau_path = ROOT / "data" / "gold" / "bureau"
    if gold_bureau_path.exists():
        print(f"Reading Gold Bureau Aggregates: {gold_bureau_path}")
        df_bureau_gold = spark.read.parquet(str(gold_bureau_path))
        print(f"Gold Bureau Schema columns: {len(df_bureau_gold.columns)}")

        # Join demonstration on SK_ID_CURR
        joined = df_bronze.select("SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT").join(
            df_bureau_gold, on="SK_ID_CURR", how="inner"
        )
        print("Catalyst Physical Execution Plan for Join:")
        joined.explain(mode="simple")

        print("\nJoined Sample Results (.show(5)):")
        joined.select("SK_ID_CURR", "TARGET", "AMT_CREDIT", "BURO_RECORD_COUNT", "BURO_AMT_CREDIT_SUM_SUM").show(5)
    else:
        print(f"Note: {gold_bureau_path} not found for join demo.")

    print("\n" + "=" * 70)
    print("SPARK PROCESSING COMPLETED SUCCESSFULLY")
    print(f"SPARK UI REMAINS ONLINE AT: {ui_url}")
    print("Keep this process running to inspect Spark Jobs/Stages live in browser.")
    print("=" * 70)

    # Keep alive so reviewer can interactively inspect Spark UI in browser
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down SparkSession cleanly...")
        spark.stop()
        print("SparkSession stopped.")

if __name__ == "__main__":
    main()
