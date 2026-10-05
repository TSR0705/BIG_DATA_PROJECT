"""Reusable I/O utilities for Spark operations in BIG_DATA_PROJECT."""

from typing import Sequence
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType


def read_csv(
    spark: SparkSession,
    path: str,
    schema: StructType | None = None,
    header: bool = True,
    infer_schema: bool = False,
    **options,
) -> DataFrame:
    """Read a CSV file into a Spark DataFrame."""
    reader = spark.read.option("header", str(header).lower())
    if schema is not None:
        reader = reader.schema(schema)
    elif infer_schema:
        reader = reader.option("inferSchema", "true")

    for key, value in options.items():
        reader = reader.option(key, value)

    return reader.csv(path)


def read_parquet(spark: SparkSession, path: str, **options) -> DataFrame:
    """Read a Parquet file or directory into a Spark DataFrame."""
    reader = spark.read
    for key, value in options.items():
        reader = reader.option(key, value)
    return reader.parquet(path)


def write_parquet(
    df: DataFrame,
    path: str,
    mode: str = "overwrite",
    **options,
) -> None:
    """Write a Spark DataFrame to Parquet format."""
    writer = df.write.mode(mode)
    for key, value in options.items():
        writer = writer.option(key, value)
    writer.parquet(path)


def assert_rowcount(
    df: DataFrame,
    expected_count: int,
    table_name: str = "DataFrame",
) -> int:
    """Assert that a DataFrame has the expected number of rows."""
    actual_count = df.count()
    if actual_count != expected_count:
        raise AssertionError(
            f"Row count mismatch for {table_name}: expected {expected_count}, got {actual_count}"
        )
    return actual_count


def assert_unique(
    df: DataFrame,
    key_columns: Sequence[str],
    table_name: str = "DataFrame",
) -> None:
    """Assert that key_columns form a unique key with no nulls."""
    total_count = df.count()
    distinct_count = df.select(*key_columns).distinct().count()
    if distinct_count != total_count:
        duplicates = total_count - distinct_count
        raise AssertionError(
            f"Uniqueness violation on {table_name} for keys {list(key_columns)}: "
            f"{duplicates} duplicate rows found ({distinct_count} distinct vs {total_count} total)"
        )
