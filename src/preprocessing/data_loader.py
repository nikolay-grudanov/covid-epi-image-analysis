"""
Data loading module for medical datasets using PySpark.

This module provides functions to load medical datasets from various formats
including CSV, Parquet, JSON, and other PySpark-supported formats.
"""

from typing import Optional
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType


def load_dataset(
    spark: SparkSession,
    file_path: str,
    format: str = "csv",
    header: bool = True,
    infer_schema: bool = True,
    schema: Optional[StructType] = None,
    mode: str = "PERMISSIVE",
    **kwargs,
) -> DataFrame:
    """
    Load a medical dataset from file into a PySpark DataFrame.

    Parameters:
    -----------
    spark : SparkSession
        Active Spark session
    file_path : str
        Path to the dataset file
    format : str, default "csv"
        File format (csv, parquet, json, orc, avro, etc.)
    header : bool, default True
        Whether the file has a header row
    infer_schema : bool, default True
        Whether to infer schema automatically (ignored if schema is provided)
    schema : StructType, optional
        Explicit schema definition (recommended for CSV files to avoid warnings)
    mode : str, default "PERMISSIVE"
        Error handling mode (PERMISSIVE, DROPMALFORMED, FAILFAST)
    **kwargs : dict
        Additional parameters passed to Spark reader

    Returns:
    --------
    DataFrame
        Loaded PySpark DataFrame

    Example:
    --------
    >>> df = load_dataset(spark, "data/raw/metadata.csv")
    >>> df.show(5)

    >>> # Using explicit schema (recommended):
    >>> from pyspark.sql.types import StructType, StructField, StringType
    >>> schema = StructType([StructField("patientid", StringType(), True)])
    >>> df = load_dataset(spark, "data/raw/metadata.csv", schema=schema)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if format == "csv":
        # Use explicit schema if provided, otherwise infer schema
        df = spark.read.csv(
            file_path,
            header=header,
            schema=schema,
            inferSchema=infer_schema if schema is None else False,
            mode=mode,
            ignoreTrailingWhiteSpace=True,
            **kwargs,
        )
    elif format == "parquet":
        df = spark.read.parquet(file_path, **kwargs)
    elif format == "json":
        df = spark.read.json(file_path, **kwargs)
    elif format == "orc":
        df = spark.read.orc(file_path, **kwargs)
    elif format == "avro":
        df = spark.read.format("avro").load(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return df
