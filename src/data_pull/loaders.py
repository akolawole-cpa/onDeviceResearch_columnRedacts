"""
Data Loaders Module

Functions for loading data from Delta tables and processing wonky study UUIDs.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from typing import List, Optional, Dict
import pandas as pd


def load_delta_table(
    spark: SparkSession,
    path: str,
    filters: Optional[Dict[str, str]] = None
):
    """
    Load a Delta table with optional filters.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    path : str
        Path to the Delta table
    filters : dict, optional
        Dictionary of column: value filters to apply
        
    Returns:
    --------
    DataFrame
        Spark DataFrame
    """
    df = spark.read.format("delta").load(path)
    
    if filters:
        for column, value in filters.items():
            if isinstance(value, str):
                df = df.filter(col(column) == value)
            elif isinstance(value, tuple) and len(value) == 2:
                # Handle range filters (min, max)
                if value[0] is not None:
                    df = df.filter(col(column) >= value[0])
                if value[1] is not None:
                    df = df.filter(col(column) <= value[1])
    
    return df


def load_user_table(
    spark: SparkSession,
    silver_path: str,
    country: str = "GB"
):
    """
    Load user table filtered by country.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    silver_path : str
        Base path to silver layer
    country : str
        Country filter (default: "GB")
        
    Returns:
    --------
    DataFrame
        Spark DataFrame with user data
    """
    return load_delta_table(
        spark,
        f"{silver_path}user",
        filters={"country": country}
    )


def load_task_complete_table(
    spark: SparkSession,
    silver_path: str,
    min_date: str = "2025-10-10",
    task_origin: str = "odr"
):
    """
    Load task_complete table with date and origin filters.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    silver_path : str
        Base path to silver layer
    min_date : str
        Minimum date filter (YYYY-MM-DD format)
    task_origin : str
        Task origin filter (default: "odr")
        
    Returns:
    --------
    DataFrame
        Spark DataFrame with task completion data
    """
    from pyspark.sql.functions import col as F_col
    
    df = spark.read.format("delta").load(f"{silver_path}task_complete")
    df = df.filter(F_col("date_completed") >= min_date)
    df = df.filter(F_col("taskOrigin") == task_origin)
    
    return df


def load_respondent_info_table(
    spark: SparkSession,
    silver_path: str,
    country: str = "GB"
):
    """
    Load respondent_info table with profile columns expanded.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    silver_path : str
        Base path to silver layer
    country : str
        Country filter (default: "GB")
        
    Returns:
    --------
    DataFrame
        Spark DataFrame with respondent info and profile data
    """
    df = load_delta_table(
        spark,
        f"{silver_path}respondent_info",
        filters={"country": country}
    )
    
    df = df.select("respondent_pk", "profile.*")
    
    return df


def load_task_table(
    spark: SparkSession,
    silver_path: str,
    origin: str = "odr"
):
    """
    Load task table with column renaming.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    silver_path : str
        Base path to silver layer
    origin : str
        Task origin filter (default: "odr")
        
    Returns:
    --------
    DataFrame
        Spark DataFrame with task data
    """
    df = load_delta_table(
        spark,
        f"{silver_path}task",
        filters={"origin": origin}
    )
    
    # Rename columns to avoid conflicts
    df = (
        df.withColumnRenamed("pk", "task_pk")
        .withColumnRenamed("origin_id", "task_origin_id")
        .withColumnRenamed("targeting_type", "task_targeting_type")
        .withColumnRenamed("length_of_task", "task_length_of_task")
        .withColumnRenamed("country", "task_country")
        .withColumnRenamed("category", "task_category")
        .withColumnRenamed("points", "task_points")
    )
    
    return df


def load_wonky_study_balance(
    spark: SparkSession,
    uuid: str,
    base_path: str = "/mnt/project-repository-prod",
    cols_to_include: Optional[List[str]] = None
):
    """
    Load balance table for a specific wonky study UUID.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    uuid : str
        Study UUID
    base_path : str
        Base path to project repository
    cols_to_include : list, optional
        List of columns to include in the subset
        
    Returns:
    --------
    DataFrame
        Spark DataFrame with balance data for the study
    """
    try:
        balance = spark.read.format("delta").load(
            f"{base_path}/{uuid}/final-data/balance"
        )
        
        balance_subset = balance.filter(col("match_id_type") == "respondent_pk")
        
        if cols_to_include:
            available_cols = [c for c in cols_to_include if c in balance_subset.columns]
            balance_subset = balance_subset.select(*available_cols)
        
        balance_subset = balance_subset.withColumn("uuid", lit(uuid))
        
        return balance_subset
        
    except Exception as e:
        print(f"WARNING: Failed to load UUID {uuid}: {str(e)[:100]}")
        return None


def load_all_wonky_studies(
    spark: SparkSession,
    uuids: List[str],
    base_path: str = "/mnt/project-repository-prod",
    cols_to_include: Optional[List[str]] = None,
    verbose: bool = True
):
    """
    Load balance tables for all wonky study UUIDs.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    uuids : List[str]
        List of study UUIDs
    base_path : str
        Base path to project repository
    cols_to_include : list, optional
        List of columns to include
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    tuple
        (list of DataFrames, list of failed UUIDs)
    """
    balance_dfs = []
    failed_uuids = []
    
    for i, uuid in enumerate(uuids, 1):
        balance_df = load_wonky_study_balance(
            spark,
            uuid,
            base_path,
            cols_to_include
        )
        
        if balance_df is not None:
            balance_dfs.append(balance_df)
        else:
            failed_uuids.append(uuid)
        
        if verbose and i % 10 == 0:
            print(f"  Processed {i}/{len(uuids)} studies...")
    
    if verbose and len(uuids) % 10 != 0:
        print(f"  Processed {len(uuids)}/{len(uuids)} studies...")
    
    return balance_dfs, failed_uuids

