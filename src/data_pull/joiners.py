"""
Data Joiners Module

Functions for joining multiple Spark DataFrames.
"""

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
    # Join user with task_complete
    joined = user_df.join(
        task_complete_df,
        user_df.respondent_pk == task_complete_df.respondentPk,
        join_type
    )
    
    # Join with respondent_info
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
    left_on: list = ["respondentPk", "taskPk"],
    right_on: list = ["balance_respondentPk", "task_pk"],
    how: str = "left"
):
    """
    Merge wonky study data with user info DataFrame (pandas).
    
    Parameters:
    -----------
    user_info_df : pd.DataFrame
        User info DataFrame (pandas)
    wonky_respondent_df : pd.DataFrame
        Wonky respondent DataFrame (pandas)
    left_on : list
        Column names for left DataFrame
    right_on : list
        Column names for right DataFrame
    how : str
        Type of merge (default: "left")
        
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame
    """
    import pandas as pd
    
    merged = user_info_df.merge(
        wonky_respondent_df,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffixes=('', '_wonky'),
        indicator=True
    )
    
    return merged

