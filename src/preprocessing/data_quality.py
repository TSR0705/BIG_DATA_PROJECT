"""Data quality profiling utility for PySpark DataFrames.

Performs efficient table-level and column-level quality checks, null analysis,
distinct counts, numeric summaries, and anomaly scans using Spark aggregations.
"""

from datetime import datetime, timezone
import math
from typing import Any, Sequence
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import (
    ByteType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NumericType,
    ShortType,
    StringType,
)


def is_numeric_type(dtype) -> bool:
    """Check if a Spark DataType is numeric."""
    return isinstance(
        dtype,
        (
            NumericType,
            IntegerType,
            LongType,
            ShortType,
            ByteType,
            DoubleType,
            FloatType,
            DecimalType,
        ),
    )


def sanitize_val(val: Any) -> Any:
    """Ensure numeric values are JSON-serializable (replace NaN/inf with None)."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


def profile_table(
    df: DataFrame,
    table_name: str,
    source_path: str,
    key_columns: Sequence[str] | None = None,
    candidate_grain: Sequence[str] | None = None,
    max_top_categories: int = 5,
    approx_distinct_rsd: float = 0.03,
) -> dict[str, Any]:
    """Profile a Spark DataFrame returning a serializable profiling dictionary."""
    timestamp = datetime.now(timezone.utc).isoformat()
    total_rows = df.count()
    cols = df.columns
    total_cols = len(cols)
    schema = df.schema

    profile: dict[str, Any] = {
        "table": table_name,
        "source": source_path,
        "rows": total_rows,
        "columns": total_cols,
        "column_names": cols,
        "profiling_timestamp": timestamp,
        "primary_key": None,
        "candidate_grain": None,
        "columns_profile": [],
    }

    if total_rows == 0:
        return profile

    # 1. Primary Key Audit
    if key_columns:
        pk_cols = [c for c in key_columns if c in cols]
        if pk_cols:
            null_cond = F.coalesce(*[F.col(c).isNull().cast("int") for c in pk_cols])
            pk_nulls = df.filter(null_cond > 0).count()
            pk_distinct = df.select(*pk_cols).distinct().count()
            pk_duplicates = total_rows - pk_distinct
            profile["primary_key"] = {
                "columns": pk_cols,
                "total_rows": total_rows,
                "nulls": pk_nulls,
                "distinct": pk_distinct,
                "duplicates": pk_duplicates,
                "unique": (pk_duplicates == 0 and pk_nulls == 0),
            }

    # 2. Candidate Grain Audit (for history/monthly tables)
    if candidate_grain:
        cg_cols = [c for c in candidate_grain if c in cols]
        if cg_cols:
            cg_distinct = df.select(*cg_cols).distinct().count()
            cg_duplicates = total_rows - cg_distinct
            profile["candidate_grain"] = {
                "columns": cg_cols,
                "total_rows": total_rows,
                "distinct": cg_distinct,
                "duplicates": cg_duplicates,
                "unique": (cg_duplicates == 0),
            }

    # 3. Batched column statistics
    # Separate numeric and string columns
    numeric_cols = [f.name for f in schema.fields if is_numeric_type(f.dataType)]
    string_cols = [f.name for f in schema.fields if isinstance(f.dataType, StringType)]

    # Aggregation Pass 1: Null counts, XNA counts, and distinct counts
    # For large tables (>1M rows), use approx_count_distinct for non-key columns
    agg_exprs = []
    for c in cols:
        agg_exprs.append(F.count(F.when(F.col(c).isNull(), 1)).alias(f"__null__{c}"))
        if total_rows > 1_000_000:
            agg_exprs.append(
                F.approx_count_distinct(F.col(c), rsd=approx_distinct_rsd).alias(f"__distinct__{c}")
            )
        else:
            agg_exprs.append(F.countDistinct(F.col(c)).alias(f"__distinct__{c}"))

    for c in string_cols:
        agg_exprs.append(
            F.count(F.when(F.col(c) == "XNA", 1)).alias(f"__xna__{c}")
        )

    # Execute pass 1
    pass1_row = df.select(agg_exprs).first().asDict()

    # Aggregation Pass 2: Numeric statistics (min, max, mean, stddev) in chunks of 40 columns
    numeric_stats: dict[str, dict[str, Any]] = {}
    chunk_size = 40
    for i in range(0, len(numeric_cols), chunk_size):
        chunk = numeric_cols[i : i + chunk_size]
        num_exprs = []
        for c in chunk:
            num_exprs.extend([
                F.min(F.col(c)).alias(f"__min__{c}"),
                F.max(F.col(c)).alias(f"__max__{c}"),
                F.mean(F.col(c)).alias(f"__mean__{c}"),
                F.stddev(F.col(c)).alias(f"__std__{c}"),
            ])
        chunk_row = df.select(num_exprs).first().asDict()
        for c in chunk:
            numeric_stats[c] = {
                "min": sanitize_val(chunk_row.get(f"__min__{c}")),
                "max": sanitize_val(chunk_row.get(f"__max__{c}")),
                "mean": sanitize_val(chunk_row.get(f"__mean__{c}")),
                "stddev": sanitize_val(chunk_row.get(f"__std__{c}")),
            }

    # Build column profiles
    for field in schema.fields:
        c = field.name
        null_cnt = pass1_row.get(f"__null__{c}", 0) or 0
        null_pct = round((null_cnt / total_rows) * 100, 4) if total_rows > 0 else 0.0
        distinct_cnt = pass1_row.get(f"__distinct__{c}", 0) or 0

        col_dict: dict[str, Any] = {
            "name": c,
            "spark_type": str(field.dataType),
            "nullable": field.nullable,
            "null_count": int(null_cnt),
            "null_pct": float(null_pct),
            "distinct_count": int(distinct_cnt),
            "is_distinct_approx": (total_rows > 1_000_000),
        }

        if c in numeric_stats:
            col_dict.update(numeric_stats[c])

        if c in string_cols:
            xna_cnt = pass1_row.get(f"__xna__{c}", 0) or 0
            xna_pct = round((xna_cnt / total_rows) * 100, 4) if total_rows > 0 else 0.0
            col_dict["xna_count"] = int(xna_cnt)
            col_dict["xna_pct"] = float(xna_pct)

            # If distinct count is small (< 30), fetch top frequent categories
            if distinct_cnt <= 30 and total_rows < 5_000_000:
                try:
                    top_vals = (
                        df.filter(F.col(c).isNotNull())
                        .groupBy(c)
                        .count()
                        .orderBy(F.desc("count"))
                        .limit(max_top_categories)
                        .collect()
                    )
                    col_dict["top_values"] = [
                        {"value": str(r[c]), "count": int(r["count"])} for r in top_vals
                    ]
                except Exception:
                    col_dict["top_values"] = []

        profile["columns_profile"].append(col_dict)

    return profile
