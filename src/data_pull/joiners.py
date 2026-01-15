"""
Data Joiners Module

Functions for joining multiple Spark DataFrames.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from typing import Optional, List
import pandas as pd


def join_user_task_respondent(
    user_df: DataFrame,
    task_complete_df: DataFrame,
    respondent_info_df: DataFrame,
    ditr_df: Optional[DataFrame] = None,
    join_type: str = "inner"
) -> DataFrame:
    """
    Join user, task_complete, respondent_info, and optionally DITR tables.
    
    This creates a task-level dataset with user demographics, profile info,
    and optionally device/hardware information.
    
    Parameters
    ----------
    user_df : DataFrame
        User DataFrame (from load_user_table)
    task_complete_df : DataFrame
        Task complete DataFrame (from load_task_complete_table)
    respondent_info_df : DataFrame
        Respondent info DataFrame (from load_respondent_info_table)
    ditr_df : DataFrame, optional
        Device info DataFrame (from load_ditr_table). If provided, joins
        hardware, manufacturer, OS, and app_version columns.
    join_type : str
        Type of join (default: "inner")
        
    Returns
    -------
    DataFrame
        Joined DataFrame at task-completion level
    """
    # User -> Task Complete
    joined = user_df.join(
        task_complete_df,
        user_df.respondent_pk == task_complete_df.respondentPk,
        join_type
    )
    
    # -> Respondent Info (drop duplicate respondent_pk after join)
    joined = joined.join(
        respondent_info_df,
        user_df.respondent_pk == respondent_info_df.respondent_pk,
        join_type
    ).drop(respondent_info_df.respondent_pk)
    
    # -> DITR (device info)
    if ditr_df is not None:
        ditr_cols_to_rename = ['date_created', 'respondent_pk', 'hardware',
                               'manufacturer', 'os']
        for col_name in ditr_cols_to_rename:
            if col_name in ditr_df.columns:
                ditr_df = ditr_df.withColumnRenamed(col_name, f"ditr_{col_name}")
        
        joined = joined.join(
            ditr_df,
            user_df.respondent_pk == ditr_df.ditr_respondent_pk,
            "left"
        )
    
    return joined


def join_wonky_balance_with_task(
    balance_df: DataFrame,
    task_df: DataFrame,
    balance_survey_col: str = "survey_pk",
    task_origin_id_col: str = "task_origin_id",
    join_type: str = "inner"
) -> DataFrame:
    """
    Join wonky study balance table with task table.
    
    Parameters
    ----------
    balance_df : DataFrame
        Balance DataFrame from wonky study
    task_df : DataFrame
        Task DataFrame
    balance_survey_col : str
        Survey column name in balance DataFrame
    task_origin_id_col : str
        Origin ID column name in task DataFrame
    join_type : str
        Type of join (default: "inner")
        
    Returns
    -------
    DataFrame
        Joined DataFrame
    """
    return balance_df.join(
        task_df,
        balance_df[balance_survey_col] == task_df[task_origin_id_col],
        join_type
    )


def merge_wonky_data_with_user_info(
    user_info_df: pd.DataFrame,
    wonky_respondent_df: pd.DataFrame,
    how: str = "left"
) -> pd.DataFrame:
    """
    Merge wonky study data with user info DataFrame.

    Creates a composite key (respondent + task) to merge at task level,
    and adds a wonky_user_flag for respondent-level identification.

    Parameters
    ----------
    user_info_df : pd.DataFrame
        User info DataFrame (task-level)
    wonky_respondent_df : pd.DataFrame
        Wonky respondent DataFrame (task-level aggregated)
    how : str
        Type of merge (default: "left")

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with wonky_user_flag and wonky_study_count
    """
    user_info_df = user_info_df.copy()
    wonky_respondent_df = wonky_respondent_df.copy()
    
    user_info_df['user_task_pk'] = (
        user_info_df['respondentPk'].astype(str) + '_' +
        user_info_df['taskPk'].astype(str)
    )
    wonky_respondent_df['user_task_pk'] = (
        wonky_respondent_df['balance_respondentPk'].astype(str) + '_' +
        wonky_respondent_df['task_pk'].astype(str)
    )

    wonky_respondent_set = set(wonky_respondent_df['balance_respondentPk'].unique())
    user_info_df['wonky_user_flag'] = (
        user_info_df['respondentPk'].isin(wonky_respondent_set).astype(int)
    )

    merge_cols = ['user_task_pk', 'task_pk', 'balance_respondentPk', 'wonky_study_count',
                  'request-remote-addr', 'task_targeting_type', 'exposure_band']
    available_merge_cols = [c for c in merge_cols if c in wonky_respondent_df.columns]

    return user_info_df.merge(
        wonky_respondent_df[available_merge_cols],
        on='user_task_pk',
        how=how,
        suffixes=('', '_wonky'),
        indicator=True
    )


def drop_duplicate_columns(df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate columns from a pandas DataFrame.
    
    This is useful after merges that create columns like 'task_pk' and 'task_pk_wonky',
    or when Spark-to-pandas conversion results in duplicate column names.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing duplicate columns
    keep : str
        Which duplicate to keep: 'first' or 'last'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate columns removed
        
    Example
    -------
    >>> df = drop_duplicate_columns(user_info_df)
    """
    return df.loc[:, ~df.columns.duplicated(keep=keep)]


def merge_task_metadata(
    user_info_df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    task_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Merge task metadata into user_info DataFrame, handling duplicates cleanly.
    
    This merges on taskPk -> task_pk and drops the redundant task_pk column
    to avoid duplicate column issues.
    
    Parameters
    ----------
    user_info_df : pd.DataFrame
        User info DataFrame with 'taskPk' column
    tasks_df : pd.DataFrame
        Tasks DataFrame with 'task_pk' and other metadata columns
    task_cols : list, optional
        Columns to bring from tasks_df. Defaults to ['task_pk', 'task_length_of_task']
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with task metadata added
        
    Example
    -------
    >>> user_info_df = merge_task_metadata(user_info_df, tasks_df)
    """
    if task_cols is None:
        task_cols = ['task_pk', 'task_length_of_task']
    
    # Ensure task_pk is in the list (needed for merge)
    if 'task_pk' not in task_cols:
        task_cols = ['task_pk'] + task_cols
    
    # Only select columns that exist
    available_cols = [c for c in task_cols if c in tasks_df.columns]
    
    merged = user_info_df.merge(
        tasks_df[available_cols],
        left_on='taskPk',
        right_on='task_pk',
        how='left'
    )
    
    # Drop the redundant task_pk (we already have taskPk)
    if 'task_pk' in merged.columns and 'taskPk' in merged.columns:
        merged = merged.drop(columns=['task_pk'])
    
    return merged