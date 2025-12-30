"""
Data cleaning module for medical datasets.

This module provides functions to clean and standardize medical data,
including removing duplicates, handling outliers, and standardizing categorical values.
"""

from typing import Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_data(
    df: DataFrame,
    remove_duplicates: bool = True,
    remove_outliers: Optional[dict] = None,
    standardize_categories: Optional[dict] = None,
) -> DataFrame:
    """
    Clean and preprocess medical data.

    Parameters:
    -----------
    df : DataFrame
        Input PySpark DataFrame
    remove_duplicates : bool, default True
        Whether to remove duplicate rows
    remove_outliers : dict, optional
        Dictionary mapping column names to (lower_bound, upper_bound) tuples
        for outlier removal
    standardize_categories : dict, optional
        Dictionary mapping column names to standardization rules
        Example: {"finding": {"COVID": "COVID-19", "Pneumonia": "Pneumonia"}}

    Returns:
    --------
    DataFrame
        Cleaned PySpark DataFrame

    Example:
    --------
    >>> df_cleaned = clean_data(df, remove_duplicates=True)
    >>> df_cleaned = clean_data(
    ...     df,
    ...     standardize_categories={
    ...         "finding": {
    ...             lambda x: "COVID" in x: "COVID-19",
    ...             lambda x: "Pneumonia" in x: "Pneumonia"
    ...         }
    ...     }
    ... )
    """
    cleaned_df = df

    # Remove duplicates
    if remove_duplicates:
        initial_count = cleaned_df.count()
        cleaned_df = cleaned_df.dropDuplicates()
        final_count = cleaned_df.count()
        print(f"Removed {initial_count - final_count} duplicate rows")

    # Remove outliers
    if remove_outliers:
        for column, (lower_bound, upper_bound) in remove_outliers.items():
            initial_count = cleaned_df.count()
            cleaned_df = cleaned_df.filter(
                (F.col(column) >= lower_bound) & (F.col(column) <= upper_bound)
            )
            final_count = cleaned_df.count()
            print(f"Removed {initial_count - final_count} outliers from {column}")

    # Standardize categories
    if standardize_categories:
        for column, rules in standardize_categories.items():
            if column in cleaned_df.columns:
                for pattern, standardized_value in rules.items():
                    if callable(pattern):
                        cleaned_df = cleaned_df.withColumn(
                            column,
                            F.when(
                                pattern(F.col(column)), standardized_value
                            ).otherwise(F.col(column)),
                        )
                    else:
                        cleaned_df = cleaned_df.withColumn(
                            column,
                            F.when(
                                F.col(column) == pattern, standardized_value
                            ).otherwise(F.col(column)),
                        )

    return cleaned_df
