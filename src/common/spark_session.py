"""Spark session factory for BIG_DATA_PROJECT.

Loads configs/spark.yaml and initializes a local SparkSession configured
for high-performance data processing on Windows.
"""

import os
import pathlib
import sys
import yaml
from pyspark.sql import SparkSession

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_spark_config() -> dict:
    cfg_file = ROOT / "configs" / "spark.yaml"
    if cfg_file.exists():
        with open(cfg_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def setup_spark_environment(cfg: dict) -> None:
    java_home = cfg.get("java_home") or os.environ.get("JAVA_HOME")
    hadoop_home = cfg.get("hadoop_home") or os.environ.get("HADOOP_HOME", r"C:\hadoop")

    if java_home and os.path.isdir(str(java_home)):
        os.environ["JAVA_HOME"] = str(java_home)
        java_bin = os.path.join(str(java_home), "bin")
        if java_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")

    if hadoop_home and os.path.isdir(str(hadoop_home)):
        os.environ["HADOOP_HOME"] = str(hadoop_home)
        hadoop_bin = os.path.join(str(hadoop_home), "bin")
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark(app_name: str | None = None) -> SparkSession:
    """Create and return a configured SparkSession."""
    cfg = load_spark_config()
    setup_spark_environment(cfg)

    app = app_name or cfg.get("app_name", "LoanDefaultPrediction-Phase1-Profiling")
    master = cfg.get("master", "local[*]")
    driver_mem = cfg.get("driver_memory", "8g")
    shuffle_parts = str(cfg.get("shuffle_partitions", 32))
    adaptive = str(cfg.get("adaptive_enabled", True)).lower()
    driver_host = cfg.get("driver_host", "127.0.0.1")

    spark = (
        SparkSession.builder
        .appName(app)
        .master(master)
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.shuffle.partitions", shuffle_parts)
        .config("spark.sql.adaptive.enabled", adaptive)
        .config("spark.driver.host", driver_host)
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
