"""
Data Joiners Module.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, concat_ws, when, broadcast
from typing import Optional, List
import pandas as pd


def join_user_task_respondent(
    user_df: DataFrame,
    task_complete_df: DataFrame,
    respondent_info_df: DataFrame,
    ditr_df: Optional[DataFrame] = None,
    join_type: str = "inner",
) -> DataFrame:
    """
    Join user, task_complete, respondent_info, and optionally DITR tables.
    """
    joined = user_df.join(
        task_complete_df,
        user_df.respondent_pk == task_complete_df.respondentPk,
        join_type,
    )

    joined = joined.join(
        respondent_info_df,
        user_df.respondent_pk == respondent_info_df.respondent_pk,
        join_type,
    ).drop(respondent_info_df.respondent_pk)

    if ditr_df is not None:
        ditr_cols_to_rename = [
            "date_created",
            "respondent_pk",
            "hardware",
            "manufacturer",
            "os",
        ]
        for col_name in ditr_cols_to_rename:
            if col_name in ditr_df.columns:
                ditr_df = ditr_df.withColumnRenamed(col_name, f"ditr_{col_name}")

        joined = joined.join(
            ditr_df, user_df.respondent_pk == ditr_df.ditr_respondent_pk, "left"
        )

    return joined


def join_wonky_balance_with_task(
    balance_df: DataFrame,
    task_df: DataFrame,
    balance_survey_col: str = "survey_pk",
    task_origin_id_col: str = "task_origin_id",
    join_type: str = "inner",
) -> DataFrame:
    """Join wonky study balance table with task table."""
    return balance_df.join(
        task_df,
        balance_df[balance_survey_col] == task_df[task_origin_id_col],
        join_type,
    )


def join_stripe_verification(
    main_df: DataFrame,
    stripe_df: DataFrame,
    main_join_col: str = "respondent_pk",
    stripe_join_col: str = "respondent_pk",
    country_alias: str = "stripe_country",
) -> DataFrame:
    """
    Join stripe verification data to main dataframe.

    Adds stripe_country column from stripe_verification table.
    Uses left join to preserve all records (stripe_country can be null).

    Parameters
    ----------
    main_df : DataFrame
        Main Spark DataFrame to join to
    stripe_df : DataFrame
        Stripe verification Spark DataFrame
    main_join_col : str
        Column name in main_df to join on
    stripe_join_col : str
        Column name in stripe_df to join on
    country_alias : str
        Alias for the country column in output

    Returns
    -------
    DataFrame
        Joined DataFrame with stripe_country column added
    """
    # Select only needed columns and alias country to avoid conflicts
    stripe_cols = stripe_df.select(
        col(stripe_join_col).alias("_stripe_respondent_pk"),
        col("country").alias(country_alias),
    )

    # Left join to preserve all records
    result_df = main_df.join(
        stripe_cols,
        main_df[main_join_col] == stripe_cols["_stripe_respondent_pk"],
        "left",
    ).drop("_stripe_respondent_pk")

    return result_df


def merge_wonky_data_spark(
    user_info_spark: DataFrame,
    wonky_respondent_spark: DataFrame,
    user_respondent_col: str = "respondentPk",
    user_task_col: str = "taskPk",
    wonky_respondent_col: str = "balance_respondentPk",
    wonky_task_col: str = "task_pk",
) -> DataFrame:
    """
    Merge wonky study data with user info DataFrame in spark.
    """
    user_info_spark = user_info_spark.withColumn(
        "user_task_pk",
        concat_ws(
            "_",
            col(user_respondent_col).cast("string"),
            col(user_task_col).cast("string"),
        ),
    )

    wonky_respondent_spark = wonky_respondent_spark.withColumn(
        "user_task_pk",
        concat_ws(
            "_",
            col(wonky_respondent_col).cast("string"),
            col(wonky_task_col).cast("string"),
        ),
    )

    wonky_respondents = wonky_respondent_spark.select(wonky_respondent_col).distinct()

    user_info_spark = (
        user_info_spark.join(
            broadcast(wonky_respondents.withColumn("_is_wonky_user", lit(1))),
            user_info_spark[user_respondent_col]
            == wonky_respondents[wonky_respondent_col],
            "left",
        )
        .withColumn("wonky_user_flag", when(col("_is_wonky_user") == 1, 1).otherwise(0))
        .drop("_is_wonky_user")
    )

    if wonky_respondent_col in user_info_spark.columns:
        col_count = sum(1 for c in user_info_spark.columns if c == wonky_respondent_col)
        if col_count > 1:
            user_info_spark = user_info_spark.drop(
                wonky_respondents[wonky_respondent_col]
            )

    wonky_merge_cols = [
        "user_task_pk",
        "task_pk",
        "balance_respondentPk",
        "wonky_study_count",
        "request-remote-addr",
        "task_targeting_type",
        "exposure_band",
        "survey_type",
    ]

    available_cols = [
        c for c in wonky_merge_cols if c in wonky_respondent_spark.columns
    ]

    wonky_subset = wonky_respondent_spark.select(*available_cols)

    for c in available_cols:
        if c != "user_task_pk":
            if c in user_info_spark.columns:
                wonky_subset = wonky_subset.withColumnRenamed(c, f"{c}_wonky")

    result = user_info_spark.join(
        broadcast(wonky_subset), on="user_task_pk", how="left"
    )

    indicator_col = None
    for c in available_cols:
        if c != "user_task_pk":
            check_col = f"{c}_wonky" if f"{c}_wonky" in result.columns else c
            if check_col in result.columns:
                indicator_col = check_col
                break

    if indicator_col:
        result = result.withColumn(
            "_merge",
            when(col(indicator_col).isNotNull(), lit("both")).otherwise(
                lit("left_only")
            ),
        )
    else:
        result = result.withColumn("_merge", lit("left_only"))

    wonky_count_col = (
        "wonky_study_count_wonky"
        if "wonky_study_count_wonky" in result.columns
        else "wonky_study_count"
    )

    if wonky_count_col in result.columns:
        result = result.withColumn(
            "wonky_study_count",
            when(col(wonky_count_col).isNotNull(), col(wonky_count_col)).otherwise(
                lit(0)
            ),
        )
        if wonky_count_col == "wonky_study_count_wonky":
            result = result.drop("wonky_study_count_wonky")
    else:
        result = result.withColumn("wonky_study_count", lit(0))

    return result


def prepare_final_output(
    merged_spark: DataFrame,
    columns_to_keep: Optional[List[str]] = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Convert final Spark DataFrame to pandas with optional column selection.
    """

    original_cols = merged_spark.columns
    seen = {}
    new_cols = []

    for col_name in original_cols:
        if col_name in seen:
            seen[col_name] += 1
            new_name = f"{col_name}_dup{seen[col_name]}"
            new_cols.append(new_name)
        else:
            seen[col_name] = 0
            new_cols.append(col_name)

    if new_cols != original_cols:
        merged_spark = merged_spark.toDF(*new_cols)
        print(
            f"  Note: Renamed {sum(1 for c in new_cols if '_dup' in c)} duplicate columns"
        )

    if columns_to_keep:
        available = [c for c in columns_to_keep if c in merged_spark.columns]

        missing = [c for c in columns_to_keep if c not in merged_spark.columns]
        if missing:
            print(
                f"  Warning: Requested columns not found: {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

        if available:
            merged_spark = merged_spark.select(*available)
        else:
            print("  Warning: No requested columns found, returning all columns")

    pdf = merged_spark.toPandas()

    if drop_duplicates:
        pdf = pdf.loc[:, ~pdf.columns.duplicated(keep="first")]

    return pdf


def deduplicate_spark_columns(spark_df: DataFrame) -> DataFrame:
    """
    Deduplicate column names in a Spark DataFrame.
    """
    original_cols = spark_df.columns
    seen = {}
    new_cols = []

    for col_name in original_cols:
        if col_name in seen:
            seen[col_name] += 1
            new_cols.append(f"{col_name}_dup{seen[col_name]}")
            print(f"Renamed column {col_name} to {new_cols[-1]}")
        else:
            seen[col_name] = 0
            new_cols.append(col_name)

    if new_cols != original_cols:
        return spark_df.toDF(*new_cols)

    return spark_df


def validate_merge_equivalence(
    spark_result_pdf: pd.DataFrame, original_columns: List[str] = None
) -> dict:
    """
    Validate that Spark merge output has expected columns and structure.
    """
    if original_columns is None:
        original_columns = [
            "user_task_pk",
            "wonky_user_flag",
            "wonky_study_count",
            "_merge",
        ]

    results = {"passed": True, "missing_cols": [], "extra_cols": [], "checks": {}}

    for col in original_columns:
        if col not in spark_result_pdf.columns:
            results["missing_cols"].append(col)
            results["passed"] = False

    if "_merge" in spark_result_pdf.columns:
        merge_values = set(spark_result_pdf["_merge"].unique())
        expected_values = {"both", "left_only"}
        results["checks"]["_merge_values"] = merge_values.issubset(
            expected_values | {None}
        )
        if not results["checks"]["_merge_values"]:
            results["passed"] = False

    if "wonky_study_count" in spark_result_pdf.columns:
        null_count = spark_result_pdf["wonky_study_count"].isna().sum()
        results["checks"]["wonky_study_count_no_nulls"] = null_count == 0
        if null_count > 0:
            results["passed"] = False
            results["checks"]["wonky_study_count_null_count"] = int(null_count)

    if "wonky_user_flag" in spark_result_pdf.columns:
        unique_flags = set(spark_result_pdf["wonky_user_flag"].unique())
        results["checks"]["wonky_user_flag_binary"] = unique_flags.issubset({0, 1})

    return results
