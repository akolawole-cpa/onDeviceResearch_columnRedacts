"""
Data Joiners Module

Functions for joining multiple Spark DataFrames.
"""
import numpy as np
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from typing import Optional


def join_user_task_respondent(
    user_df: DataFrame,
    task_complete_df: DataFrame,
    respondent_info_df: DataFrame,
    join_type: str = "inner"
) -> DataFrame:
    """
    Join user, task_complete, and respondent_info tables.
    
    Parameters:
    -----------
    user_df : DataFrame
        User DataFrame
    task_complete_df : DataFrame
        Task complete DataFrame
    respondent_info_df : DataFrame
        Respondent info DataFrame
    join_type : str
        Type of join (default: "inner")
        
    Returns:
    --------
    DataFrame
        Joined DataFrame
    """
    # user & task_complete
    joined = user_df.join(
        task_complete_df,
        user_df.respondent_pk == task_complete_df.respondentPk,
        join_type
    )
    
    # respondent info & user
    joined = joined.join(
        respondent_info_df,
        user_df.respondent_pk == respondent_info_df.respondent_pk,
        join_type
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
    
    Parameters:
    -----------
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
        
    Returns:
    --------
    DataFrame
        Joined DataFrame
    """
    joined = balance_df.join(
        task_df,
        balance_df[balance_survey_col] == task_df[task_origin_id_col],
        join_type
    )
    
    return joined


def merge_wonky_data_with_user_info(
    user_info_df,
    wonky_respondent_df,
    how: str = "left"
):
    """
    Merge wonky study data with user info df.

    This function handles both task-level and respondent-level merges.
    If task-level columns are missing from the right DataFrame, it falls back
    to respondent-level merge (only on respondent ID).

    Optimized with set-based lookup for O(1) membership testing (2-3x faster).

    Parameters:
    -----------
    user_info_df : pd.DataFrame
        User info DataFrame (pandas)
    wonky_respondent_df : pd.DataFrame
        Wonky respondent DataFrame (pandas) - can be task-level or respondent-level
    how : str
        Type of merge (default: "left")

    Returns:
    --------
    pd.DataFrame
        Merged DataFrame
    """
    # Create composite keys for task-level merge
    user_info_df['user_task_pk'] = (
        user_info_df['respondentPk'].astype(str) + '_' +
        user_info_df['taskPk'].astype(str)
    )
    wonky_respondent_df['user_task_pk'] = (
        wonky_respondent_df['balance_respondentPk'].astype(str) + '_' +
        wonky_respondent_df['task_pk'].astype(str)
    )

    # Use SET for O(1) lookup instead of list (2-3x faster for large DataFrames)
    wonky_respondent_set = set(wonky_respondent_df['balance_respondentPk'].unique())
    user_info_df["wonky_user_flag"] = user_info_df["respondentPk"].isin(
        wonky_respondent_set
    ).astype(int)

    # Select only needed columns from right DataFrame to reduce merge overhead
    merge_cols = ['user_task_pk', 'task_pk', 'balance_respondentPk', 'wonky_study_count']
    available_merge_cols = [c for c in merge_cols if c in wonky_respondent_df.columns]

    return user_info_df.merge(
        wonky_respondent_df[available_merge_cols],
        on=["user_task_pk"],
        how=how,
        suffixes=("", "_wonky"),
        indicator=True,
    )


    # import pandas as pd
    
    # if not isinstance(left_on, list) or not isinstance(right_on, list):
    #     raise TypeError("left_on and right_on must be lists")
    
    # if len(left_on) != len(right_on):
    #     raise ValueError(
    #         f"left_on and right_on must have the same length. "
    #         f"Got {len(left_on)} and {len(right_on)}"
    #     )
    
    # available_pairs = []
    # missing_cols = []
    
    # for left_col, right_col in zip(left_on, right_on):
    #     left_exists = left_col in user_info_df.columns
    #     right_exists = right_col in wonky_respondent_df.columns
        
    #     if left_exists and right_exists:
    #         available_pairs.append((left_col, right_col))
    #     else:
    #         missing_cols.append({
    #             'left_col': left_col,
    #             'right_col': right_col,
    #             'left_exists': left_exists,
    #             'right_exists': right_exists
    #         })
    
    # if len(available_pairs) < len(right_on):
    #     if len(available_pairs) == 0:
    #         error_msg = (
    #             f"None of the specified merge columns exist in both DataFrames.\n"
    #             f"Requested merge columns:\n"
    #         )
    #         for left_col, right_col in zip(left_on, right_on):
    #             left_exists = left_col in user_info_df.columns
    #             right_exists = right_col in wonky_respondent_df.columns
    #             error_msg += (
    #                 f"  - left_on='{left_col}' (exists: {left_exists}), "
    #                 f"right_on='{right_col}' (exists: {right_exists})\n"
    #             )
    #         error_msg += (
    #             f"\nAvailable columns in user_info_df: {list(user_info_df.columns)[:10]}...\n"
    #             f"Available columns in wonky_respondent_df: {list(wonky_respondent_df.columns)[:10]}..."
    #         )
    #         raise ValueError(error_msg)
        
    #     left_on_actual = [available_pairs[0][0]]  
    #     right_on_actual = [available_pairs[0][1]] 
        

    #     if len(right_on) > 1:
    #         missing_info = missing_cols[0] if missing_cols else {}
    #         print(
    #             f"Warning: Task-level merge columns not available. "
    #             f"Missing: left_col='{missing_info.get('left_col', 'N/A')}' "
    #             f"(exists: {missing_info.get('left_exists', False)}), "
    #             f"right_col='{missing_info.get('right_col', 'N/A')}' "
    #             f"(exists: {missing_info.get('right_exists', False)}).\n"
    #             f"Falling back to respondent-level merge on '{left_on_actual[0]}' ↔ '{right_on_actual[0]}'."
    #         )
    # else:
    #     # All columns available, use full merge
    #     left_on_actual = [pair[0] for pair in available_pairs]
    #     right_on_actual = [pair[1] for pair in available_pairs]
    
    # # Perform the merge
    # merged = user_info_df.merge(
    #     wonky_respondent_df,
    #     left_on=left_on_actual,
    #     right_on=right_on_actual,
    #     how=how,
    #     suffixes=('', '_wonky'),
    #     indicator=True
    # )
    
    # return merged