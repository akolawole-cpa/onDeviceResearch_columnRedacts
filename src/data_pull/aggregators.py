"""
Data Aggregators Module

Functions for aggregating data at different levels.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import count, unix_timestamp, col
from pyspark.sql.window import Window
from functools import reduce
from typing import List, Optional
import pandas as pd


def enrich_user_info_with_task_counts(
    user_info_df: DataFrame,
    respondent_id_col: str = "respondentPk",
    task_id_col: str = "taskPk",
    date_completed_col: str = "date_completed",
    date_created_col: str = "date_created",
) -> DataFrame:
    """
    Enrich user info DataFrame with task counts and time calculations.

    Adds:
    - task_time_taken_s: Time between task creation and completion (seconds)
    - task_completed: Total tasks completed by this respondent

    Parameters
    ----------
    user_info_df : DataFrame
        User info DataFrame (Spark)
    respondent_id_col : str
        Name of respondent ID column
    task_id_col : str
        Name of task ID column
    date_completed_col : str
        Name of date completed column
    date_created_col : str
        Name of date created column

    Returns
    -------
    DataFrame
        Enriched DataFrame
    """
    respondent_window = Window.partitionBy(respondent_id_col)

    return (
        user_info_df
        .withColumn(
            "task_time_taken_s",
            unix_timestamp(col(date_completed_col)) - unix_timestamp(col(date_created_col))
        )
        .withColumn(
            "task_completed", 
            count(col(task_id_col)).over(respondent_window)
        )
    )


def union_wonky_study_dataframes(balance_dfs: List[DataFrame]) -> DataFrame:
    """
    Union multiple wonky study balance DataFrames.

    Parameters
    ----------
    balance_dfs : List[DataFrame]
        List of balance DataFrames

    Returns
    -------
    DataFrame
        Unioned DataFrame
        
    Raises
    ------
    ValueError
        If no DataFrames provided
    """
    if not balance_dfs:
        raise ValueError("No DataFrames provided")

    return reduce(lambda df1, df2: df1.union(df2), balance_dfs)


def create_wonky_respondent_summary(
    wonky_respondent_df: pd.DataFrame,
    respondent_id_col: str = "respondent_pk",
    categorical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create summary statistics from wonky respondent DataFrame.

    Parameters
    ----------
    wonky_respondent_df : pd.DataFrame
        Wonky respondents DataFrame (task-level)
    respondent_id_col : str
        Name of respondent ID column
    categorical_cols : List[str], optional
        Categorical columns to include in grouping

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with aggregated metrics
    """
    if categorical_cols is None:
        categorical_cols = [
            'platform_name', 'hardware_version', 'yob', 
            'survey_locale', 'exposure_band'
        ]

    # Build aggregation dict
    agg_dict = {"task_pk": ["count", "nunique"]}
    
    if "uuid" in wonky_respondent_df.columns:
        agg_dict["uuid"] = "nunique"
    elif "wonky_study_count" in wonky_respondent_df.columns:
        agg_dict["wonky_study_count"] = "sum"

    # Filter to available categorical columns
    available_cats = [c for c in categorical_cols if c in wonky_respondent_df.columns]
    grouping_cols = [respondent_id_col] + available_cats

    summary = (
        wonky_respondent_df
        .groupby(grouping_cols)
        .agg(agg_dict)
        .reset_index()
    )

    # Flatten multi-level column names
    summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) and col[1] else col[0]
        for col in summary.columns.values
    ]

    # Standardize column names
    rename_map = {
        respondent_id_col: "balance_respondentPk",
        "uuid_nunique": "total_wonky_studies",
        "wonky_study_count_sum": "total_wonky_studies"
    }
    rename_map = {k: v for k, v in rename_map.items() if k in summary.columns}
    
    return summary.rename(columns=rename_map)


def calculate_wonky_task_ratio(
    task_completed_df: pd.DataFrame,
    wonky_summary_df: pd.DataFrame,
    task_completed_col: str = "task_completed",
    wonky_instances_col: str = "total_wonky_studies",
    respondent_id_col: str = "respondentPk_tc"
) -> pd.DataFrame:
    """
    Calculate wonky task ratio for each respondent.

    The ratio represents what fraction of a respondent's completed tasks
    were "wonky" tasks.

    Parameters
    ----------
    task_completed_df : pd.DataFrame
        DataFrame with task completion counts per respondent
    wonky_summary_df : pd.DataFrame
        DataFrame with wonky study summary per respondent
    task_completed_col : str
        Name of total tasks column
    wonky_instances_col : str
        Name of wonky task instances column
    respondent_id_col : str
        Name of respondent ID column

    Returns
    -------
    pd.DataFrame
        DataFrame with wonky_task_ratio column added
    """
    wonky_counts = task_completed_df.merge(
        wonky_summary_df,
        left_on=respondent_id_col,
        right_on="balance_respondentPk",
        how="inner"
    )

    wonky_counts["wonky_task_ratio"] = (
        wonky_counts[wonky_instances_col] / wonky_counts[task_completed_col]
    )

    return wonky_counts