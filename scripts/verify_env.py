"""Phase 0 Environment Verifier for Loan Default Prediction Using Big Data Analytics.

Validates the native Windows development environment:
1. Python version and executable
2. PySpark import and version
3. Java / JDK 17 availability
4. Windows Hadoop support (HADOOP_HOME, winutils.exe, hadoop.dll)
5. SparkSession creation with master local[*]
6. 1,000,000 synthetic rows generation
7. Transformation and groupBy (shuffle)
8. Parquet write
9. Parquet read-back and count verification
10. Automatic cleanup of temporary verification directory
11. Confirmation message: ENVIRONMENT VERIFIED
"""

import os
import pathlib
import shutil
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# ----------------------------------------------------------------------
# 1. Configuration & Runtime Environment Pinning
# ----------------------------------------------------------------------
def setup_environment() -> None:
    try:
        import yaml
        cfg_file = ROOT / "configs" / "spark.yaml"
        if cfg_file.exists():
            with open(cfg_file, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
    except Exception:
        cfg = {}

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

    # Ensure worker processes use the venv Python
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


setup_environment()

print(f"Python executable : {sys.executable}")
print(f"Python version    : {sys.version.split()[0]}")
print(f"JAVA_HOME         : {os.environ.get('JAVA_HOME')}")
print(f"HADOOP_HOME       : {os.environ.get('HADOOP_HOME')}")

# Check Hadoop Windows binaries
hadoop_bin = pathlib.Path(os.environ.get("HADOOP_HOME", r"C:\hadoop")) / "bin"
winutils_path = hadoop_bin / "winutils.exe"
hadoop_dll_path = hadoop_bin / "hadoop.dll"

if not winutils_path.exists() or not hadoop_dll_path.exists():
    print(
        "WARNING: Hadoop Windows native binaries check:\n"
        f"  winutils.exe exists: {winutils_path.exists()} ({winutils_path})\n"
        f"  hadoop.dll exists  : {hadoop_dll_path.exists()} ({hadoop_dll_path})\n"
        "If Parquet write fails, verify HADOOP_HOME, winutils.exe, and hadoop.dll."
    )

# ----------------------------------------------------------------------
# 2. PySpark Import & Execution Check
# ----------------------------------------------------------------------
spark = None
temp_dir = ROOT / "data" / "_phase0_spark_check"

try:
    import pyspark
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

    print(f"PySpark version   : {pyspark.__version__}")

    # Build local Spark session
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("LoanDefaultPrediction-Phase0-Verification")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    print("SparkSession created successfully.")

    # Step 4-6: Generate 1,000,000 synthetic rows and group by
    print("Generating 1,000,000 synthetic rows...")
    df = spark.range(1_000_000)

    print("Executing groupBy and shuffle...")
    grouped = (
        df.withColumn("group_id", F.col("id") % 100)
          .groupBy("group_id")
          .count()
    )

    # Force execution with an action
    collected = grouped.collect()
    num_groups = len(collected)
    total_rows = sum(r["count"] for r in collected)

    assert num_groups == 100, f"Expected 100 groups, got {num_groups}"
    assert total_rows == 1_000_000, f"Expected total 1,000,000 rows, got {total_rows}"
    assert all(r["count"] == 10_000 for r in collected), "Group counts are inconsistent"
    print(f"groupBy verification: {num_groups} groups, {total_rows:,} rows counted. OK.")

    # Step 7: Parquet write
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    output_path = str(temp_dir)
    print(f"Writing Parquet to temporary directory: {output_path}...")
    grouped.write.mode("overwrite").parquet(output_path)
    print("Parquet write completed.")

    # Step 8: Parquet read
    print("Reading Parquet back...")
    read_back = spark.read.parquet(output_path)
    read_count = read_back.count()
    assert read_count == 100, f"Expected 100 rows from Parquet, got {read_count}"
    print(f"Parquet read verified: {read_count} rows read back. OK.")

    # Step 9: Cleanup
    print("Cleaning up temporary verification directory...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    if not temp_dir.exists():
        print("Cleanup successful.")

    print("\nENVIRONMENT VERIFIED")

except Exception as exc:
    print("\nENVIRONMENT VERIFICATION FAILED")
    print(f"Error: {exc}")
    print("\nPlease verify:")
    print("  1. JAVA_HOME points to a valid JDK 17 (e.g., C:\\Java\\jdk-17.0.20.1+1)")
    print("  2. HADOOP_HOME points to C:\\hadoop")
    print("  3. C:\\hadoop\\bin\\winutils.exe exists")
    print("  4. C:\\hadoop\\bin\\hadoop.dll exists")
    traceback.print_exc()
    sys.exit(1)

finally:
    if spark is not None:
        try:
            spark.stop()
        except Exception:
            pass
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
