"""
Missing values handling module for medical datasets.

This module provides functions to handle missing values in medical data,
including filling with statistics, dropping, and imputation strategies.
"""

from typing import Optional, Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType


def handle_missing_values(
    df: DataFrame,
    strategy: str = "median",
    fill_values: Optional[Dict[str, Any]] = None,
    drop_threshold: Optional[float] = None,
) -> DataFrame:
    """
    Handle missing values in medical data.

    Parameters:
    -----------
    df : DataFrame
        Input PySpark DataFrame
    strategy : str, default "median"
        Strategy for filling missing numeric values
        Options: "median", "mean", "mode", "zero", "drop"
    fill_values : dict, optional
        Dictionary mapping column names to specific fill values
        Overrides automatic strategy for specified columns
    drop_threshold : float, optional
        If provided, drop rows with more than this fraction of missing values
        (e.g., 0.5 means drop rows with >50% missing values)

    Returns:
    --------
    DataFrame
        DataFrame with missing values handled

    Example:
    --------
    >>> df_filled = handle_missing_values(df, strategy="median")
    >>> df_filled = handle_missing_values(
    ...     df,
    ...     fill_values={"age": 50, "sex": "Unknown"}
    ... )
    >>> df_filled = handle_missing_values(df, drop_threshold=0.3)
    """
    filled_df = df

    # Drop rows with too many missing values if threshold specified
    if drop_threshold is not None and 0 <= drop_threshold <= 1:
        total_columns = len(filled_df.columns)
        null_counts = sum(
            [F.when(F.col(c).isNull(), 1).otherwise(0) for c in filled_df.columns]
        )
        filled_df = filled_df.filter((null_counts / total_columns) <= drop_threshold)
        print(f"Dropped rows with >{drop_threshold*100}% missing values")

    # Apply fill values if provided
    if fill_values:
        filled_df = filled_df.na.fill(fill_values)
        print(
            f"Filled missing values with custom values for columns: {list(fill_values.keys())}"
        )

    # Handle missing values based on strategy
    if strategy == "drop":
        initial_count = filled_df.count()
        filled_df = filled_df.na.drop()
        final_count = filled_df.count()
        print(f"Dropped {initial_count - final_count} rows with missing values")

    elif strategy == "zero":
        # Fill numeric columns with 0
        numeric_columns = [
            field.name
            for field in filled_df.schema.fields
            if isinstance(field.dataType, NumericType)
        ]
        if numeric_columns:
            filled_df = filled_df.na.fill({col: 0 for col in numeric_columns})
            print(
                f"Filled numeric missing values with 0 for columns: {numeric_columns}"
            )

    elif strategy in ["median", "mean", "mode"]:
        # Calculate and fill with statistics for numeric columns
        numeric_columns = [
            field.name
            for field in filled_df.schema.fields
            if isinstance(field.dataType, NumericType)
        ]

        for column in numeric_columns:
            if strategy == "median":
                stat_value = filled_df.select(F.median(column)).collect()[0][0]
            elif strategy == "mean":
                stat_value = filled_df.select(F.mean(column)).collect()[0][0]
            elif strategy == "mode":
                stat_value = (
                    filled_df.groupBy(column)
                    .count()
                    .orderBy(F.desc("count"))
                    .first()[0]
                )

            if stat_value is not None:
                filled_df = filled_df.na.fill({column: stat_value})
                print(f"Filled {column} missing values with {strategy}: {stat_value}")

    return filled_df
