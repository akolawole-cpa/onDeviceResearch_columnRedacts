"""
Data Loaders Module

Functions for loading data from Delta tables and processing wonky study UUIDs.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, row_number
from pyspark.sql.window import Window
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def load_delta_table(
    spark: SparkSession,
    path: str,
    filters: Optional[Dict[str, str]] = None
) -> DataFrame:
    """Load a Delta table with optional filters."""
    df = spark.read.format("delta").load(path)

    if filters:
        for column, value in filters.items():
            if isinstance(value, str):
                df = df.filter(col(column) == value)
            elif isinstance(value, tuple) and len(value) == 2:
                if value[0] is not None:
                    df = df.filter(col(column) >= value[0])
                if value[1] is not None:
                    df = df.filter(col(column) <= value[1])

    return df


def load_user_table(
    spark: SparkSession, silver_path: str, country: str = "GB"
) -> DataFrame:
    """Load user table filtered by country."""
    return load_delta_table(spark, f"{silver_path}user", filters={"country": country})


def load_task_complete_table(
    spark: SparkSession,
    silver_path: str,
    min_date: str = "2025-10-10",
    task_origin: str = "odr",
) -> DataFrame:
    """Load task_complete table with date and origin filters."""
    df = spark.read.format("delta").load(f"{silver_path}task_complete")
    return df.filter(
        (col("date_completed") >= min_date) & (col("taskOrigin") == task_origin)
    )


def load_respondent_info_table(
    spark: SparkSession,
    silver_path: str,
    country: str = "GB",
    exclude_cols: Optional[List[str]] = None,
) -> DataFrame:
    """Load respondent_info table with profile columns expanded."""
    df = load_delta_table(
        spark, f"{silver_path}respondent_info", filters={"country": country}
    )
    df = df.select("respondent_pk", "profile.*")

    if exclude_cols:
        cols_to_keep = [c for c in df.columns if c not in exclude_cols]
        df = df.select(*cols_to_keep)

    return df


def load_task_table(
    spark: SparkSession, silver_path: str, origin: str = "odr"
) -> DataFrame:
    """Load task table with column renaming to avoid conflicts."""
    df = load_delta_table(spark, f"{silver_path}task", filters={"origin": origin})

    renames = {
        "pk": "task_pk",
        "origin_id": "task_origin_id",
        "targeting_type": "task_targeting_type",
        "length_of_task": "task_length_of_task",
        "country": "task_country",
        "category": "task_category",
        "points": "task_points",
    }

    for old_name, new_name in renames.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)

    return df


def load_stripe_verification(
    spark: SparkSession,
    silver_path: str,
    select_cols: Optional[List[str]] = None,
) -> DataFrame:
    """
    Load stripe_verification table from silver layer.

    Parameters
    ----------
    spark : SparkSession
    silver_path : str
        Path to silver layer
    select_cols : list, optional
        Columns to select. Defaults to ['respondent_pk', 'country']

    Returns
    -------
    DataFrame
        Stripe verification data with country information
    """
    if select_cols is None:
        select_cols = ["respondent_pk", "country"]

    df = spark.read.format("delta").load(f"{silver_path}stripe_verification")

    available_cols = [c for c in select_cols if c in df.columns]
    if available_cols:
        df = df.select(*available_cols)

    return df


def load_ditr_table(
    spark: SparkSession,
    silver_path: str,
    user_df: Optional[DataFrame] = None,
    select_cols: Optional[List[str]] = None,
) -> DataFrame:
    """
    Load device_id_to_respondent (DITR) table with latest record per respondent.

    Optional user data load (user_df) to filter for countries.
    """
    if select_cols is None:
        select_cols = [
            "respondent_pk",
            "date_created",
            "limit_ad_tracking",
            "app_version",
            "hardware",
            "manufacturer",
            "os",
            "os_version",
        ]

    ditr = spark.read.format("delta").load(f"{silver_path}device_id_to_respondent")

    window_spec = Window.partitionBy("respondent_pk").orderBy(
        col("date_created").desc()
    )
    latest_ditr = (
        ditr.withColumn("_row_num", row_number().over(window_spec))
        .filter(col("_row_num") == 1)
        .drop("_row_num")
    )

    if user_df is not None:
        latest_ditr = user_df.select("respondent_pk").join(
            latest_ditr, "respondent_pk", "left"
        )

    available_cols = [c for c in select_cols if c in latest_ditr.columns]
    return latest_ditr.select(*available_cols)


def load_wonky_study_balance(
    spark: SparkSession,
    uuid: str,
    base_path: str = "/mnt/project-repository-prod",
    cols_to_include: Optional[List[str]] = None,
) -> Optional[DataFrame]:
    """
    Load balance table for a specific wonky study UUID.

    Parameters
    ----------
    spark : SparkSession
    uuid : str
        Study UUID
    base_path : str
        Base path to project repository
    cols_to_include : list, optional
        Columns to include in the subset

    Returns
    -------
    DataFrame or None
        Balance data for the study, or None if loading fails
    """
    try:
        balance = spark.read.format("delta").load(
            f"{base_path}/{uuid}/final-data/balance"
        )

        if "match_id_type" in balance.columns:
            balance_subset = balance.filter(col("match_id_type") == "respondent_pk")
        else:
            print(
                f"INFO: UUID {uuid} has no 'match_id_type' column - adding placeholder"
            )
            balance_subset = balance.withColumn("match_id_type", lit("temp_match_id"))

        if cols_to_include:
            available_cols = [c for c in cols_to_include if c in balance_subset.columns]
            balance_subset = balance_subset.select(*available_cols)

        return balance_subset.withColumn("uuid", lit(uuid))

    except Exception as e:
        print(f"WARNING: Failed to load UUID {uuid}: {str(e)[:100]}")
        return None


def load_all_wonky_studies(
    spark: SparkSession,
    uuids: List[str],
    base_path: str = "/mnt/project-repository-prod",
    cols_to_include: Optional[List[str]] = None,
    verbose: bool = True,
    max_workers: int = 8,
    parallel: bool = True,
) -> Tuple[List[DataFrame], List[str]]:
    """
    Load balance tables for all wonky study UUIDs.

    Uses parallel loading with ThreadPoolExecutor for improved performance.

    Parameters
    ----------
    spark : SparkSession
    uuids : List[str]
        List of study UUIDs
    base_path : str
        Base path to project repository
    cols_to_include : list, optional
        Columns to include
    verbose : bool
        Whether to print progress
    max_workers : int
        Maximum parallel workers (default: 8)
    parallel : bool
        If True, use parallel loading

    Returns
    -------
    Tuple[List[DataFrame], List[str]]
        (list of DataFrames, list of failed UUIDs)
    """
    if not uuids:
        return [], []

    balance_dfs = []
    failed_uuids = []

    if not parallel:
        # Sequential fallback >> a lot slower.
        for i, uuid in enumerate(uuids, 1):
            df = load_wonky_study_balance(spark, uuid, base_path, cols_to_include)
            if df is not None:
                balance_dfs.append(df)
            else:
                failed_uuids.append(uuid)

            if verbose and i % 10 == 0:
                print(f"  Processed {i}/{len(uuids)} studies...")

        return balance_dfs, failed_uuids

    # Parallel loading
    lock = threading.Lock()

    def load_single(uuid: str):
        return uuid, load_wonky_study_balance(spark, uuid, base_path, cols_to_include)

    if verbose:
        print(f"  Loading {len(uuids)} studies with {max_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single, uuid): uuid for uuid in uuids}
        completed = 0

        for future in as_completed(futures):
            try:
                uuid, df = future.result()
                with lock:
                    if df is not None:
                        balance_dfs.append(df)
                    else:
                        failed_uuids.append(uuid)
                    completed += 1

                if verbose and completed % 10 == 0:
                    print(f"  Processed {completed}/{len(uuids)} studies...")

            except Exception as e:
                uuid = futures[future]
                with lock:
                    failed_uuids.append(uuid)
                    completed += 1
                if verbose:
                    print(f"  WARNING: Exception loading {uuid}: {str(e)[:50]}")

    if verbose:
        print(f"  Successfully loaded: {len(balance_dfs)}, Failed: {len(failed_uuids)}")

    return balance_dfs, failed_uuids
