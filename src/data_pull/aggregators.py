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
    
    # Fix: Use col() to reference columns by name, as unix_timestamp() expects Column objects
    # Verify that date_completed_col and date_created_col exist in the DataFrame
    enriched = (
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


def _get_mode_value(series):
    """Helper function to get mode value from a series."""
    mode_values = series.mode()
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    else:
        return series.iloc[0] if len(series) > 0 else None


def create_wonky_respondent_summary(
    wonky_respondent_df: pd.DataFrame,
    respondent_id_col: str = "respondent_pk"
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
    # Build aggregation dictionary dynamically based on available columns
    agg_dict = {
        'task_pk': ['count', 'nunique'],
    }
    
    # Count unique UUIDs (studies) per respondent if uuid column exists
    if 'uuid' in wonky_respondent_df.columns:
        agg_dict['uuid'] = 'nunique'  # Count unique studies
    elif 'wonky_study_count' in wonky_respondent_df.columns:
        agg_dict['wonky_study_count'] = 'sum'
    
    # Add mode aggregations for categorical columns (only if they exist)
    # For mode, we'll aggregate first, then calculate mode separately to avoid naming issues
    categorical_cols = ['survey_pk', 'platform_name', 'hardware_version', 
                       'yob', 'survey_locale', 'exposure_band']
    
    # Store which categorical columns we're aggregating
    categorical_cols_to_process = [col for col in categorical_cols if col in wonky_respondent_df.columns]
    
    # For now, use 'first' as a proxy (we'll calculate mode after if needed)
    # Or we can use the helper function - pandas should preserve column names
    for col in categorical_cols_to_process:
        agg_dict[col] = _get_mode_value
    
    summary = (
        wonky_respondent_df.groupby(respondent_id_col)
        .agg(agg_dict)
        .reset_index()
    )
    
    # Flatten column names
    summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) and col[1] else col[0]
        for col in summary.columns.values
    ]
    
    # Rename columns
    rename_dict = {}
    
    # Rename respondent ID column
    if respondent_id_col in summary.columns:
        rename_dict[respondent_id_col] = 'balance_respondentPk'
    
    # Rename task columns
    if 'task_pk_count' in summary.columns:
        rename_dict['task_pk_count'] = 'wonky_task_instances'
    if 'task_pk_nunique' in summary.columns:
        rename_dict['task_pk_nunique'] = 'wonky_unique_tasks'
    
    # Rename study count column (could be uuid_nunique or wonky_study_count_sum)
    if 'uuid_nunique' in summary.columns:
        rename_dict['uuid_nunique'] = 'total_wonky_studies'
    elif 'wonky_study_count_sum' in summary.columns:
        rename_dict['wonky_study_count_sum'] = 'total_wonky_studies'
    
    # Rename categorical columns
    # After aggregation, pandas may name columns differently (e.g., 'survey_pk_get_mode_value' or just 'survey_pk')
    # We need to find and rename them to their simple names
    categorical_cols_to_rename = ['survey_pk', 'platform_name', 'hardware_version', 
                                 'yob', 'survey_locale', 'exposure_band']
    
    # First, check if columns already have simple names (pandas preserved them)
    for col in categorical_cols_to_rename:
        if col in summary.columns:
            # Column already has correct name, no rename needed
            continue
        
        # Find columns that contain this categorical column name
        # Could be 'survey_pk', 'survey_pk_get_mode_value', 'survey_pk_<lambda>', etc.
        matching_cols = [c for c in summary.columns 
                         if (c.startswith(col + '_') or c == col) 
                         and c not in rename_dict.values()]
        
        if matching_cols:
            # Use the first matching column that isn't already being renamed
            rename_dict[matching_cols[0]] = col
    
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