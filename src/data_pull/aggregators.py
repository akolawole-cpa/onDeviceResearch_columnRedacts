"""
Data Aggregators Module

Functions for aggregating data at different levels.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import count, F
from pyspark.sql.window import Window
from functools import reduce
from typing import List, Optional
import pandas as pd
import numpy as np


def enrich_user_info_with_task_counts(
    user_info_df: DataFrame,
    respondent_id_col: str = "respondentPk",
    task_id_col: str = "taskPk",
    date_completed_col: str = "date_completed",
    date_created_col: str = "date_created"
) -> DataFrame:
    """
    Enrich user info DataFrame with task counts and time calculations.
    
    Parameters:
    -----------
    user_info_df : DataFrame
        User info DataFrame
    respondent_id_col : str
        Name of respondent ID column
    task_id_col : str
        Name of task ID column
    date_completed_col : str
        Name of date completed column
    date_created_col : str
        Name of date created column
        
    Returns:
    --------
    DataFrame
        Enriched DataFrame
    """
    respondent_window = Window.partitionBy(respondent_id_col)
    
    enriched = (
        user_info_df
        .withColumn(
            "task_time_taken_s",
            F.unix_timestamp(date_completed_col) - F.unix_timestamp(date_created_col)
        )
        .withColumn(
            "task_completed",
            count(task_id_col).over(respondent_window)
        )
    )
    
    return enriched


def union_wonky_study_dataframes(
    balance_dfs: List[DataFrame]
) -> DataFrame:
    """
    Union multiple wonky study balance DataFrames.
    
    Parameters:
    -----------
    balance_dfs : List[DataFrame]
        List of balance DataFrames
        
    Returns:
    --------
    DataFrame
        Unioned DataFrame
    """
    if not balance_dfs:
        raise ValueError("No DataFrames provided")
    
    wonky_spark = reduce(lambda df1, df2: df1.union(df2), balance_dfs)
    
    return wonky_spark


def aggregate_wonky_respondent_summary(
    wonky_map_df: pd.DataFrame,
    grouping_cols: List[str],
    respondent_id_col: str = "respondent_pk"
) -> pd.DataFrame:
    """
    Aggregate wonky study data to respondent level.
    
    Parameters:
    -----------
    wonky_map_df : pd.DataFrame
        Wonky map DataFrame (pandas)
    grouping_cols : List[str]
        List of columns to group by
    respondent_id_col : str
        Name of respondent ID column
        
    Returns:
    --------
    pd.DataFrame
        Aggregated respondent-level DataFrame
    """
    available_cols = [col for col in grouping_cols if col in wonky_map_df.columns]
    
    wonky_respondent_df = (
        wonky_map_df.groupby(available_cols)
        .agg({'uuid': 'count'})
        .reset_index()
        .rename(columns={"uuid": "wonky_study_count", respondent_id_col: "balance_respondentPk"})
    )
    
    return wonky_respondent_df


def create_wonky_respondent_summary(
    wonky_respondent_df: pd.DataFrame,
    respondent_id_col: str = "balance_respondentPk"
) -> pd.DataFrame:
    """
    Create summary statistics for wonky respondents.
    
    Parameters:
    -----------
    wonky_respondent_df : pd.DataFrame
        Wonky respondent DataFrame (task-level)
    respondent_id_col : str
        Name of respondent ID column
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with aggregated metrics
    """
    summary = (
        wonky_respondent_df.groupby(respondent_id_col)
        .agg({
            'task_pk': ['count', 'nunique'],
            'wonky_study_count': 'sum',
            'survey_pk': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'platform_name': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'hardware_version': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'yob': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'survey_locale': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'exposure_band': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        })
        .reset_index()
    )
    
    # Flatten column names
    summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) and col[1] else col[0]
        for col in summary.columns.values
    ]
    
    # Rename columns
    rename_dict = {
        'balance_respondentPk': 'balance_respondentPk',
        'task_pk_count': 'wonky_task_instances',
        'task_pk_nunique': 'wonky_unique_tasks',
        'wonky_study_count_sum': 'total_wonky_studies',
        'survey_pk_<lambda>': 'survey_pk',
        'platform_name_<lambda>': 'platform_name',
        'hardware_version_<lambda>': 'hardware_version',
        'yob_<lambda>': 'yob',
        'survey_locale_<lambda>': 'survey_locale',
        'exposure_band_<lambda>': 'exposure_band',
    }
    
    # Only rename columns that exist
    rename_dict = {k: v for k, v in rename_dict.items() if k in summary.columns}
    summary = summary.rename(columns=rename_dict)
    
    return summary


def calculate_wonky_task_ratio(
    task_completed_df: pd.DataFrame,
    wonky_summary_df: pd.DataFrame,
    task_completed_col: str = "task_completed",
    wonky_instances_col: str = "wonky_task_instances",
    respondent_id_col: str = "respondentPk"
) -> pd.DataFrame:
    """
    Calculate wonky task ratio for each respondent.
    
    Parameters:
    -----------
    task_completed_df : pd.DataFrame
        DataFrame with task completion counts
    wonky_summary_df : pd.DataFrame
        DataFrame with wonky study summary
    task_completed_col : str
        Name of total tasks column
    wonky_instances_col : str
        Name of wonky task instances column
    respondent_id_col : str
        Name of respondent ID column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with wonky task ratio
    """
    wonky_counts = task_completed_df.merge(
        wonky_summary_df,
        left_on=respondent_id_col,
        right_on='balance_respondentPk',
        how='inner'
    )
    
    wonky_counts['wonky_task_ratio'] = (
        wonky_counts[wonky_instances_col] / wonky_counts[task_completed_col]
    )
    
    return wonky_counts

