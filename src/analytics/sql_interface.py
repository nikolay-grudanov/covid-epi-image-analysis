"""
SQL interface module for medical data analytics.

This module provides functions to register DataFrames as SQL tables
and execute SQL queries on medical data.
"""

from typing import Union
from pyspark.sql import DataFrame, SparkSession
import pandas as pd


def register_table(df: DataFrame, table_name: str, replace: bool = True) -> None:
    """
    Register a DataFrame as a temporary SQL table/view.

    Parameters:
    -----------
    df : DataFrame
        PySpark DataFrame to register
    table_name : str
        Name for the SQL table/view
    replace : bool, default True
        Whether to replace existing table with the same name

    Returns:
    --------
    None

    Example:
    --------
    >>> register_table(df, "patients")
    >>> df_result = spark.sql("SELECT * FROM patients")
    """
    if replace:
        df.createOrReplaceTempView(table_name)
    else:
        df.createTempView(table_name)

    print(f"Registered DataFrame as temporary view: {table_name}")


def execute_query(
    spark: SparkSession, query: str, to_pandas: bool = True, limit: int = None
) -> Union[DataFrame, pd.DataFrame]:
    """
    Execute a SQL query on registered tables.

    Parameters:
    -----------
    spark : SparkSession
        Active Spark session
    query : str
        SQL query to execute
    to_pandas : bool, default True
        Whether to convert result to pandas DataFrame
    limit : int, optional
        Limit the number of rows returned

    Returns:
    --------
    DataFrame or pandas.DataFrame
        Query result

    Example:
    --------
    >>> df = execute_query(
    ...     spark,
    ...     "SELECT finding, COUNT(*) FROM patients GROUP BY finding"
    ... )
    """
    df_result = spark.sql(query)

    if limit:
        df_result = df_result.limit(limit)

    if to_pandas:
        return df_result.toPandas()
    return df_result


def execute_standard_analytics(
    spark: SparkSession, table_name: str = "patients"
) -> dict:
    """
    Execute a standard set of analytical queries for medical data.

    Parameters:
    -----------
    spark : SparkSession
        Active Spark session
    table_name : str, default "patients"
        Name of the registered table

    Returns:
    --------
    dict
        Dictionary containing results of various analytics queries

    Example:
    --------
    >>> results = execute_standard_analytics(spark, "patients")
    >>> print(results['diagnosis_summary'])
    """
    results = {}

    # Query 1: Basic statistics by diagnosis
    query1 = f"""
    SELECT finding,
           COUNT(*) as patient_count,
           AVG(age) as avg_age,
           MIN(age) as min_age,
           MAX(age) as max_age
    FROM {table_name}
    GROUP BY finding
    ORDER BY patient_count DESC
    """
    results["diagnosis_summary"] = execute_query(spark, query1)

    # Query 2: Distribution by gender and diagnosis
    query2 = f"""
    SELECT sex,
           finding,
           COUNT(*) as count
    FROM {table_name}
    GROUP BY sex, finding
    ORDER BY sex, finding
    """
    results["gender_diagnosis"] = execute_query(spark, query2)

    # Query 3: Top N ages per diagnosis
    query3 = f"""
    SELECT finding, age, COUNT(*) as patient_count
    FROM (
        SELECT finding, age, COUNT(*) as patient_count,
               ROW_NUMBER() OVER (PARTITION BY finding ORDER BY COUNT(*) DESC) as rn
        FROM {table_name}
        GROUP BY finding, age
    )
    WHERE rn <= 5
    ORDER BY finding, patient_count DESC
    """
    results["top_ages_per_diagnosis"] = execute_query(spark, query3)

    # Query 4: Temporal trends
    query4 = f"""
    SELECT finding,
           YEAR(date) as year,
           MONTH(date) as month,
           COUNT(*) as count
    FROM {table_name}
    WHERE date IS NOT NULL
    GROUP BY finding, YEAR(date), MONTH(date)
    ORDER BY finding, year, month
    """
    results["temporal_trends"] = execute_query(spark, query4)

    # Query 5: Projection view statistics
    query5 = f"""
    SELECT view,
           finding,
           COUNT(*) as count
    FROM {table_name}
    GROUP BY view, finding
    ORDER BY view, finding
    """
    results["view_diagnosis"] = execute_query(spark, query5)

    return results
