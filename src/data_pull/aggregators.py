"""
Data Aggregators Module.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    count,
    unix_timestamp,
    col,
    sum as spark_sum,
    countDistinct,
    lit,
    first,
)
from pyspark.sql.window import Window
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

    All operations stay in Spark - no driver memory impact.
    """
    respondent_window = Window.partitionBy(respondent_id_col)

    return user_info_df.withColumn(
        "task_time_taken_s",
        unix_timestamp(col(date_completed_col)) - unix_timestamp(col(date_created_col)),
    ).withColumn("task_completed", count(col(task_id_col)).over(respondent_window))


def union_wonky_study_dataframes(balance_dfs: List[DataFrame]) -> DataFrame:
    """
    Union multiple wonky study balance DataFrames.

    Uses unionByName with allowMissingColumns to handle schema variations.
    """
    if not balance_dfs:
        raise ValueError("No DataFrames provided")

    result = balance_dfs[0]
    for df in balance_dfs[1:]:
        result = result.unionByName(df, allowMissingColumns=True)

    return result


def create_wonky_respondent_df_spark(
    wonky_map_spark: DataFrame,
    group_cols: List[str],
    cols_to_drop: Optional[List[str]] = None,
) -> DataFrame:
    """Aggregate wonky map to respondent-task level in spark."""
    # Filter to columns that exist
    available_group_cols = [c for c in group_cols if c in wonky_map_spark.columns]

    # Aggregate
    agg_exprs = [
        count("uuid").alias("balance_study_count"),
        spark_sum("wonky_study_count").alias("wonky_study_count"),
    ]

    result = wonky_map_spark.groupBy(*available_group_cols).agg(*agg_exprs)

    if "respondent_pk" in result.columns:
        result = result.withColumnRenamed("respondent_pk", "balance_respondentPk")

    if cols_to_drop:
        cols_to_actually_drop = [c for c in cols_to_drop if c in result.columns]
        result = result.drop(*cols_to_actually_drop)

    return result


def create_wonky_respondent_summary_spark(
    wonky_respondent_spark: DataFrame,
    respondent_id_col: str = "balance_respondentPk",
    categorical_cols: Optional[List[str]] = None,
) -> DataFrame:
    """Create summary statistics from wonky respondent DataFrame in spark."""
    if categorical_cols is None:
        categorical_cols = [
            "platform_name",
            "hardware_version",
            "yob",
            "survey_locale",
            "exposure_band",
            "survey_type",
        ]

    available_cats = [
        c for c in categorical_cols if c in wonky_respondent_spark.columns
    ]
    grouping_cols = [respondent_id_col] + available_cats

    agg_exprs = [
        count("task_pk").alias("task_pk_count"),
        countDistinct("task_pk").alias("task_pk_nunique"),
    ]

    if "uuid" in wonky_respondent_spark.columns:
        agg_exprs.append(countDistinct("uuid").alias("total_wonky_studies"))
    elif "wonky_study_count" in wonky_respondent_spark.columns:
        agg_exprs.append(spark_sum("wonky_study_count").alias("total_wonky_studies"))

    return wonky_respondent_spark.groupBy(*grouping_cols).agg(*agg_exprs)


def calculate_wonky_task_ratio_spark(
    user_info_spark: DataFrame,
    wonky_summary_spark: DataFrame,
    user_respondent_col: str = "respondentPk",
    wonky_respondent_col: str = "balance_respondentPk",
    task_completed_col: str = "task_completed",
    wonky_instances_col: str = "total_wonky_studies",
) -> DataFrame:
    """
    Calculate wonky task ratio for each respondent in spark.

    Returns
    -------
    DataFrame
        Spark DataFrame with wonky_task_ratio column
    """
    from pyspark.sql.functions import broadcast

    respondent_tasks = user_info_spark.select(
        user_respondent_col, task_completed_col
    ).dropDuplicates([user_respondent_col])

    joined = respondent_tasks.join(
        broadcast(wonky_summary_spark),
        respondent_tasks[user_respondent_col]
        == wonky_summary_spark[wonky_respondent_col],
        "inner",
    )

    result = joined.withColumn(
        "wonky_task_ratio", col(wonky_instances_col) / col(task_completed_col)
    )

    return result
