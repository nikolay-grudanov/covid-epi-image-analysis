"""
Data quality assessment module for medical datasets.

This module provides functions to assess data quality including completeness,
consistency, validity, and accuracy metrics.
"""

from typing import Dict, List
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType


def assess_quality(df: DataFrame) -> Dict[str, any]:
    """
    Perform comprehensive data quality assessment.

    Parameters:
    -----------
    df : DataFrame
        Input PySpark DataFrame to assess

    Returns:
    --------
    dict
        Dictionary containing quality metrics:
        - total_records: Total number of records
        - total_columns: Total number of columns
        - missing_values: Dictionary of missing value counts per column
        - completeness_percentage: Overall data completeness
        - duplicate_rows: Number of duplicate rows
        - column_types: Dictionary mapping column names to data types
        - numeric_summary: Summary statistics for numeric columns

    Example:
    --------
    >>> quality_metrics = assess_quality(df)
    >>> print(f"Completeness: {quality_metrics['completeness_percentage']}%")
    """
    metrics = {}

    # Basic counts
    metrics["total_records"] = df.count()
    metrics["total_columns"] = len(df.columns)

    # Missing values per column
    missing_values = {}
    for column in df.columns:
        null_count = df.filter(F.col(column).isNull()).count()
        missing_values[column] = null_count
    metrics["missing_values"] = missing_values

    # Overall completeness percentage
    total_cells = metrics["total_records"] * metrics["total_columns"]
    missing_cells = sum(missing_values.values())
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    metrics["completeness_percentage"] = round(completeness, 2)

    # Duplicate rows
    metrics["duplicate_rows"] = df.count() - df.dropDuplicates().count()

    # Column data types
    metrics["column_types"] = {
        field.name: str(field.dataType) for field in df.schema.fields
    }

    # Numeric column summary
    numeric_columns = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, NumericType)
    ]

    numeric_summary = {}
    for column in numeric_columns:
        stats = df.select(
            F.count(column).alias("count"),
            F.mean(column).alias("mean"),
            F.stddev(column).alias("stddev"),
            F.min(column).alias("min"),
            F.max(column).alias("max"),
        ).collect()[0]

        numeric_summary[column] = {
            "count": stats["count"],
            "mean": round(stats["mean"], 2) if stats["mean"] else None,
            "stddev": round(stats["stddev"], 2) if stats["stddev"] else None,
            "min": stats["min"],
            "max": stats["max"],
        }
    metrics["numeric_summary"] = numeric_summary

    return metrics


def print_quality_report(metrics: Dict[str, any]) -> None:
    """
    Print a formatted quality assessment report.

    Parameters:
    -----------
    metrics : dict
        Quality metrics dictionary from assess_quality()
    """
    print("=" * 60)
    print("DATA QUALITY ASSESSMENT REPORT")
    print("=" * 60)

    print(f"\n📊 Total Records: {metrics['total_records']:,}")
    print(f"📋 Total Columns: {metrics['total_columns']}")
    print(f"✅ Completeness: {metrics['completeness_percentage']}%")
    print(f"🔄 Duplicate Rows: {metrics['duplicate_rows']:,}")

    print("\n--- Missing Values by Column ---")
    for column, count in metrics["missing_values"].items():
        if count > 0:
            percentage = (count / metrics["total_records"]) * 100
            print(f"{column}: {count:,} ({percentage:.2f}%)")

    if metrics["numeric_summary"]:
        print("\n--- Numeric Column Statistics ---")
        for column, stats in metrics["numeric_summary"].items():
            print(f"\n{column}:")
            print(f"  Count: {stats['count']:,}")
            print(f"  Mean: {stats['mean']}")
            print(f"  Std Dev: {stats['stddev']}")
            print(f"  Range: [{stats['min']}, {stats['max']}]")

    print("\n" + "=" * 60)
