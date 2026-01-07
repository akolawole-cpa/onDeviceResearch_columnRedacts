"""
Feature Engineering Module

Functions for creating time-based features, behavioral features, and fraud risk scores.
"""


import numpy as np
import pandas as pd
from typing import Dict, Optional


def create_task_amount_features(
    df: pd.DataFrame,
    task_col: str = "totaal_tasks_completed",
) -> pd.DataFrame:
    """
    Create task bin dummy features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing task_col
    task_col : str, default "totaal_tasks_completed"
        Name of the tasks completed column

    Returns
    -------
    pd.DataFrame
        DataFrame with added task bin dummy columns
    """
    df = df.copy()

    # Define bins and labels
    bins = [0, 10, 25, 60, 100, 150, float("inf")]
    labels = [
        "total_tasks_1_10",
        "total_tasks_11_25",
        "total_tasks_26_60",
        "total_tasks_61_100",
        "total_tasks_101_150",
        "total_tasks_151_plus",
    ]

    # Categorise
    task_bins = pd.cut(
        df[task_col],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    # One-hot encode into dummy columns
    task_dummies = pd.get_dummies(task_bins, prefix="", prefix_sep="")

    # Ensure all dummy columns exist (even if 0) and cast to int
    task_dummies = task_dummies[labels].astype(int)
    df = pd.concat([df, task_dummies], axis=1)

    return df, labels


def create_task_temporal_features(df: pd.DataFrame, date_col: str = "date_completed") -> pd.DataFrame:
    """
    Create time-related features from a datetime column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the date column
    date_col : str
        Name of the datetime column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added time features
    """
    
    df = df.copy()
    df["hour_of_day"] = df[date_col].dt.hour
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] > 5).astype(int)

    df["is_monday"]    = df[date_col].dt.day_name().eq("Monday").astype(int)
    df["is_tuesday"]   = df[date_col].dt.day_name().eq("Tuesday").astype(int)
    df["is_wednesday"] = df[date_col].dt.day_name().eq("Wednesday").astype(int)
    df["is_thursday"]  = df[date_col].dt.day_name().eq("Thursday").astype(int)
    df["is_friday"]    = df[date_col].dt.day_name().eq("Friday").astype(int)
    df["is_saturday"]  = df[date_col].dt.day_name().eq("Saturday").astype(int)
    df["is_sunday"]    = df[date_col].dt.day_name().eq("Sunday").astype(int)

    df["is_night"] = (
        (df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)
    ).astype(int)
    df["is_business_hour"] = (
        (df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 17)
    ).astype(int)
    df["is_business_hour_weekday"] = (
        (df["is_business_hour"] == 1) & (df["is_weekend"] == 0)
    ).astype(int)
    df["is_business_hour_weekend"] = (
        (df["is_business_hour"] == 1) & (df["is_weekend"] == 1)
    ).astype(int)
    
    return df


def create_all_temporal_features(
    df: pd.DataFrame, 
    date_col: str = "date_completed",
    include_hourly: bool = True
) -> pd.DataFrame:
    """
    Create comprehensive time-related features from a datetime column.
    
    Combines day-of-week, business hours, AND hourly features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the date column
    date_col : str
        Name of the datetime column
    include_hourly : bool
        If True, includes individual hour indicators (24 features)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all temporal features including:
        - hour_of_day_label: "1am", "2pm", etc.
        - hour_of_day: numeric (0-23)
        - is_hour_X features for each hour
    """
    
    df = df.copy()
    
    df["hour_of_day"] = df[date_col].dt.hour
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    def format_hour_label(hour):
        """Convert 24-hour format to 12-hour am/pm format."""
        if pd.isna(hour):
            return None
        
        hour = int(hour)
        if hour == 0:
            return "12am"
        elif hour < 12:
            return f"{hour}am"
        elif hour == 12:
            return "12pm"
        else:
            return f"{hour - 12}pm"
    
    df["hour_of_day_label"] = df["hour_of_day"].apply(format_hour_label)
    
    df["is_monday"]    = df[date_col].dt.day_name().eq("Monday").astype(int)
    df["is_tuesday"]   = df[date_col].dt.day_name().eq("Tuesday").astype(int)
    df["is_wednesday"] = df[date_col].dt.day_name().eq("Wednesday").astype(int)
    df["is_thursday"]  = df[date_col].dt.day_name().eq("Thursday").astype(int)
    df["is_friday"]    = df[date_col].dt.day_name().eq("Friday").astype(int)
    df["is_saturday"]  = df[date_col].dt.day_name().eq("Saturday").astype(int)
    df["is_sunday"]    = df[date_col].dt.day_name().eq("Sunday").astype(int)
    
    df["is_night"] = (
        (df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)
    ).astype(int)
    
    df["is_business_hour"] = (
        (df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 17)
    ).astype(int)
    
    df["is_business_hour_weekday"] = (
        (df["is_business_hour"] == 1) & (df["is_weekend"] == 0)
    ).astype(int)
    
    df["is_business_hour_weekend"] = (
        (df["is_business_hour"] == 1) & (df["is_weekend"] == 1)
    ).astype(int)
    
    if include_hourly:
        for hour in range(24):
            hour_label = format_hour_label(hour)
            df[f"is_hour_{hour}"] = (df["hour_of_day"] == hour).astype(int)
            df[f"is_{hour_label}"] = (df["hour_of_day"] == hour).astype(int)
    
    def categorize_hour_period(hour):
        if pd.isna(hour):
            return None
        elif 0 <= hour < 6:
            return "night"
        elif 6 <= hour < 9:
            return "early_morning"
        elif 9 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    df["hour_period"] = df["hour_of_day"].apply(categorize_hour_period)
    
    df["is_early_morning"] = (df["hour_period"] == "early_morning").astype(int)
    df["is_morning"] = (df["hour_period"] == "morning").astype(int)
    df["is_afternoon"] = (df["hour_period"] == "afternoon").astype(int)
    df["is_evening"] = (df["hour_period"] == "evening").astype(int)
    
    return df


def create_task_speed_features(
    df: pd.DataFrame,
    task_time_col: str = "task_time_taken_s",
    use_std_dev: bool = True,
    group_by_col: Optional[str] = None,
    min_group_size: int = 10
) -> pd.DataFrame:
    """
    Create task speed-related features using standard deviations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing task time data
    task_time_col : str
        Name of the task time column (in seconds)
    use_std_dev : bool
        If True, use standard deviations to define thresholds. If False, use legacy hardcoded thresholds.
    group_by_col : str, optional
        Column name to group by when calculating thresholds (e.g., 'task_length_of_task').
        If None, will auto-detect 'task_length_of_task' if it exists.
        If grouping column doesn't exist, uses global thresholds (backward compatible).
    min_group_size : int, default=10
        Minimum number of tasks required in a group to calculate group-specific thresholds.
        Groups with fewer tasks will use global thresholds as fallback.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added speed features:
        - is_fast: 1 std dev below mean
        - is_suspiciously_fast: 2 std dev below mean
        - is_slow: 1 std dev above mean
        - is_suspiciously_slow: 2 std dev above mean
    """
    df = df.copy()
    
    if task_time_col not in df.columns:
        raise ValueError(f"Task time column '{task_time_col}' not found in DataFrame")
    
    # Calculate task time in minutes
    df["task_time_minutes"] = df[task_time_col] / 60
    
    if use_std_dev:
        # Use percentiles instead of raw standard deviations to handle outliers robustly
        # Percentiles approximate normal distribution: 
        # - 16th percentile ≈ mean - 1σ (fast)
        # - 2.5th percentile ≈ mean - 2σ (suspiciously fast)
        # - 84th percentile ≈ mean + 1σ (slow)
        # - 97.5th percentile ≈ mean + 2σ (suspiciously slow)
        
        valid_times = df[task_time_col].dropna()
        if len(valid_times) == 0:
            raise ValueError(f"No valid values found in '{task_time_col}' column")
        
        if valid_times.nunique() == 1:
            df["is_fast"] = 0
            df["is_suspiciously_fast"] = 0
            df["is_slow"] = 0
            df["is_suspiciously_slow"] = 0
            df["is_normal_speed"] = 1 
            df["is_very_fast"] = 0
        else:
            grouping_col = None
            if group_by_col is not None:
                if group_by_col in df.columns:
                    grouping_col = group_by_col
            elif "task_length_of_task" in df.columns:
                grouping_col = "task_length_of_task"
            
            fast_threshold_global = valid_times.quantile(0.16)
            suspiciously_fast_threshold_global = valid_times.quantile(0.025)
            slow_threshold_global = valid_times.quantile(0.84)
            suspiciously_slow_threshold_global = valid_times.quantile(0.975)
            
            thresholds_dict = {}
            
            if grouping_col is not None:
                df[grouping_col + "_grouped"] = df[grouping_col].fillna("unknown")
                
                for group_value in df[grouping_col + "_grouped"].unique():
                    group_mask = df[grouping_col + "_grouped"] == group_value
                    group_times = df.loc[group_mask, task_time_col].dropna()
                    
                    if len(group_times) < min_group_size:
                        thresholds_dict[group_value] = {
                            "fast": fast_threshold_global,
                            "suspiciously_fast": suspiciously_fast_threshold_global,
                            "slow": slow_threshold_global,
                            "suspiciously_slow": suspiciously_slow_threshold_global,
                            "use_global": True
                        }
                    elif group_times.nunique() == 1:
                        thresholds_dict[group_value] = {
                            "fast": None,
                            "suspiciously_fast": None,
                            "slow": None,
                            "suspiciously_slow": None,
                            "all_same": True
                        }
                    else:
                        thresholds_dict[group_value] = {
                            "fast": group_times.quantile(0.16),
                            "suspiciously_fast": group_times.quantile(0.025),
                            "slow": group_times.quantile(0.84),
                            "suspiciously_slow": group_times.quantile(0.975),
                            "use_global": False
                        }
                
                def apply_group_thresholds(row):
                    group_val = row[grouping_col + "_grouped"]
                    thresholds = thresholds_dict.get(group_val, {
                        "fast": fast_threshold_global,
                        "suspiciously_fast": suspiciously_fast_threshold_global,
                        "slow": slow_threshold_global,
                        "suspiciously_slow": suspiciously_slow_threshold_global
                    })
                    
                    task_time = row[task_time_col]
                    
                    if thresholds.get("all_same", False):
                        return pd.Series({
                            "is_fast": 0,
                            "is_suspiciously_fast": 0,
                            "is_slow": 0,
                            "is_suspiciously_slow": 0,
                            "is_normal_speed": 1 if pd.notna(task_time) else 0
                        })
                    
                    if pd.isna(task_time):
                        return pd.Series({
                            "is_fast": 0,
                            "is_suspiciously_fast": 0,
                            "is_slow": 0,
                            "is_suspiciously_slow": 0,
                            "is_normal_speed": 0
                        })
                    
                    return pd.Series({
                        "is_fast": int(task_time < thresholds["fast"]),
                        "is_suspiciously_fast": int(task_time < thresholds["suspiciously_fast"]),
                        "is_slow": int(task_time > thresholds["slow"]),
                        "is_suspiciously_slow": int(task_time > thresholds["suspiciously_slow"]),
                        "is_normal_speed": int(
                            thresholds["fast"] <= task_time <= thresholds["slow"]
                        )
                    })
                
                speed_flags = df.apply(apply_group_thresholds, axis=1)
                df["is_fast"] = speed_flags["is_fast"]
                df["is_suspiciously_fast"] = speed_flags["is_suspiciously_fast"]
                df["is_slow"] = speed_flags["is_slow"]
                df["is_suspiciously_slow"] = speed_flags["is_suspiciously_slow"]
                df["is_normal_speed"] = speed_flags["is_normal_speed"]
                
                df = df.drop(columns=[grouping_col + "_grouped"])
            else:
                fast_threshold = fast_threshold_global
                suspiciously_fast_threshold = suspiciously_fast_threshold_global
                slow_threshold = slow_threshold_global
                suspiciously_slow_threshold = suspiciously_slow_threshold_global
                
                df["is_fast"] = (df[task_time_col] < fast_threshold).fillna(0).astype(int)
                df["is_suspiciously_fast"] = (df[task_time_col] < suspiciously_fast_threshold).fillna(0).astype(int)
                df["is_slow"] = (df[task_time_col] > slow_threshold).fillna(0).astype(int)
                df["is_suspiciously_slow"] = (df[task_time_col] > suspiciously_slow_threshold).fillna(0).astype(int)
                
                df["is_normal_speed"] = (
                    (df[task_time_col] >= fast_threshold) & 
                    (df[task_time_col] <= slow_threshold)
                ).fillna(0).astype(int)
            
            df["is_very_fast"] = df["is_fast"].copy()
        
    else:
        # Legacy hardcoded thresholds (for backward compatibility)
        suspicious_threshold = 30.0
        very_fast_threshold = 180.0
        very_slow_threshold = 3600.0
        
        df["is_suspiciously_fast"] = (df[task_time_col] < suspicious_threshold).astype(int)
        df["is_very_fast"] = (df[task_time_col] < very_fast_threshold).astype(int)
        df["is_very_slow"] = (df["task_time_minutes"] > very_slow_threshold / 60).astype(int)
    
    if "task_points" in df.columns:
        df["task_points_perMinute"] = (
            df["task_points"] / df["task_time_minutes"].replace(0, np.nan)
        )
    
    return df


def add_wonky_features(
    respondent_features: pd.DataFrame,
    wonky_studies_df: pd.DataFrame,
    respondent_id_col: str = "respondentPk"
) -> pd.DataFrame:
    """
    Add wonky study features to respondent-level DataFrame.
    
    Parameters:
    -----------
    respondent_features : pd.DataFrame
        Respondent-level features DataFrame
    wonky_studies_df : pd.DataFrame
        DataFrame with wonky study information (aggregated)
    respondent_id_col : str
        Name of the respondent ID column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added wonky features
    """
    df = respondent_features.copy()
    
    wonky_cols = ['wonky_task_instances', 'wonky_unique_tasks', 
                  'total_wonky_studies', 'wonky_task_ratio']
    available_cols = [col for col in wonky_cols if col in wonky_studies_df.columns]
    
    df = df.merge(
        wonky_studies_df[[respondent_id_col] + available_cols],
        on=respondent_id_col,
        how='left'
    )
    
    for col in available_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Wonky concentration
    if "wonky_task_ratio" in df.columns:
        print("is_high_wonky means > 50% wonky task ratio")
        df['is_high_wonky'] = (df['wonky_task_ratio'] > 0.5).astype(int)
        print("is_quite_wonky means > 30% wonky task ratio")
        df['is_quite_wonky'] = (df['wonky_task_ratio'] > 0.3).astype(int)
    
    return df


def create_fraud_risk_score(
    df: pd.DataFrame,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Create fraud risk score and tiers based on multiple behavioral indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Respondent-level DataFrame with behavioral features
    config : dict, optional
        Configuration dictionary with score weights and thresholds
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added fraud risk score and tiers
    """
    if config is None:
        config = {}
    
    df = df.copy()
    
    # Get weights from config or use defaults
    weights = config.get("fraud_score_weights", {
        "high_volume": 2,
        "suspicious_fast_rate": 2,
        "high_wonky": 2,
        "high_risk": 1,
        "low_cv": 1,
        "high_night_rate": 1,
        "high_tasks_per_day": 1
    })
    
    thresholds = config.get("fraud_score_thresholds", {
        "suspicious_fast_rate": 0.3,
        "high_risk": 0.5,
        "low_cv": 0.2,
        "night_rate": 0.5,
        "tasks_per_day": 10
    })
    
    # Quality-Risk blend flags
    if "avg_risk" in df.columns and "avg_quality" in df.columns:
        df['bad_quality_high_risk'] = (
            (df['avg_risk'] > thresholds.get("high_risk", 0.5)) & 
            (df['avg_quality'] < 0.5)
        ).astype(int)
        
        df['good_quality_low_risk'] = (
            (df['avg_risk'] < 0.3) & 
            (df['avg_quality'] > 0.7)
        ).astype(int)
    
    # Multi-factor fraud score (0-10 scale)
    fraud_score = 0
    
    if "is_high_volume" in df.columns:
        fraud_score += df['is_high_volume'] * weights["high_volume"]
    
    if "suspicious_fast_rate" in df.columns:
        fraud_score += (
            (df['suspicious_fast_rate'] > thresholds["suspicious_fast_rate"]).astype(int) * 
            weights["suspicious_fast_rate"]
        )
    
    if "is_high_wonky" in df.columns:
        fraud_score += df['is_high_wonky'].astype(int) * weights["high_wonky"]
    
    if "avg_risk" in df.columns:
        fraud_score += (
            (df['avg_risk'] > thresholds["high_risk"]).astype(int) * 
            weights["high_risk"]
        )
    
    if "task_time_cv" in df.columns:
        fraud_score += (
            (df['task_time_cv'] < thresholds["low_cv"]).astype(int) * 
            weights["low_cv"]
        )
    
    if "night_task_rate" in df.columns:
        fraud_score += (
            (df['night_task_rate'] > thresholds["night_rate"]).astype(int) * 
            weights["high_night_rate"]
        )
    
    if "tasks_per_day" in df.columns:
        fraud_score += (
            (df['tasks_per_day'] > thresholds["tasks_per_day"]).astype(int) * 
            weights["high_tasks_per_day"]
        )
    
    df['fraud_risk_score'] = fraud_score
    
    # Fraud risk tiers
    fraud_bins = config.get("fraud_score_bins", [0, 2, 4, 6, 10])
    fraud_labels = config.get("fraud_score_labels", ['Low', 'Medium', 'High', 'Critical'])
    
    df['fraud_risk_tier'] = pd.cut(
        df['fraud_risk_score'],
        bins=fraud_bins,
        labels=fraud_labels,
        include_lowest=True
    )
    
    # Suspected fraud label
    fraud_threshold = config.get("suspected_fraud_threshold", 6)
    df['suspected_fraud'] = (df['fraud_risk_score'] >= fraud_threshold).astype(int)
    
    return df


def create_wonky_risk_score(
    df: pd.DataFrame,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Create wonky risk score based on behavioral patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Respondent-level DataFrame with behavioral features
    config : dict, optional
        Configuration dictionary with score weights
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added wonky risk score
    """
    if config is None:
        config = {}
    
    df = df.copy()
    
    weights = config.get("wonky_score_weights", {
        "suspicious_fast_rate": 2,
        "high_volume": 2,
        "low_quality": 2,
        "high_risk": 1,
        "low_cv": 1,
        "high_night_rate": 1,
        "high_tasks_per_day": 1
    })
    
    thresholds = config.get("wonky_score_thresholds", {
        "suspicious_fast_rate": 0.3,
        "high_risk": 0.5,
        "low_quality": 0.5,
        "low_cv": 0.2,
        "night_rate": 0.5,
        "tasks_per_day": 10
    })
    
    # Interaction features
    if "is_high_volume" in df.columns and "suspicious_fast_rate" in df.columns:
        df['high_volume_x_suspicious_fast'] = (
            df['is_high_volume'] * 
            (df['suspicious_fast_rate'] > thresholds["suspicious_fast_rate"]).astype(int)
        ).astype(int)
    
    if "avg_quality" in df.columns and "avg_risk" in df.columns:
        df['low_quality_x_high_risk'] = (
            (df['avg_quality'] < thresholds["low_quality"]).astype(int) * 
            (df['avg_risk'] > thresholds["high_risk"]).astype(int)
        ).astype(int)
    
    if "is_high_volume" in df.columns and "avg_quality" in df.columns:
        df['high_volume_x_low_quality'] = (
            df['is_high_volume'] * 
            (df['avg_quality'] < thresholds["low_quality"]).astype(int)
        ).astype(int)
    
    # Composite wonky risk score
    wonky_score = 0
    
    if "suspicious_fast_rate" in df.columns:
        wonky_score += (
            (df['suspicious_fast_rate'] > thresholds["suspicious_fast_rate"]).astype(int) * 
            weights["suspicious_fast_rate"]
        )
    
    if "is_high_volume" in df.columns:
        wonky_score += df['is_high_volume'].astype(int) * weights["high_volume"]
    
    if "avg_quality" in df.columns:
        wonky_score += (
            (df['avg_quality'] < thresholds["low_quality"]).astype(int) * 
            weights["low_quality"]
        )
    
    if "avg_risk" in df.columns:
        wonky_score += (
            (df['avg_risk'] > thresholds["high_risk"]).astype(int) * 
            weights["high_risk"]
        )
    
    if "task_time_cv" in df.columns:
        wonky_score += (
            (df['task_time_cv'] < thresholds["low_cv"]).astype(int) * 
            weights["low_cv"]
        )
    
    if "night_task_rate" in df.columns:
        wonky_score += (
            (df['night_task_rate'] > thresholds["night_rate"]).astype(int) * 
            weights["high_night_rate"]
        )
    
    if "tasks_per_day" in df.columns:
        wonky_score += (
            (df['tasks_per_day'] > thresholds["tasks_per_day"]).astype(int) * 
            weights["high_tasks_per_day"]
        )
    
    df['wonky_risk_score'] = wonky_score
    
    # Wonky risk tiers
    wonky_bins = config.get("wonky_score_bins", [0, 2, 4, 6, 10])
    wonky_labels = config.get("wonky_score_labels", ['Low', 'Medium', 'High', 'Critical'])
    
    df['wonky_risk_tier'] = pd.cut(
        df['wonky_risk_score'],
        bins=wonky_bins,
        labels=wonky_labels,
        include_lowest=True
    )
    
    return df


def add_rating_delta(
    df: pd.DataFrame, feature: str, delta_period: int | str = 1
) -> pd.DataFrame:
    """
    Add a '{rating}_delta' column representing the change in {rating}
    from a previous task to the current task for each respondent.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['respondentPk', 'date_completed', f'{rating}'].
    delta_period : int or 'max', default 1
        - int (1, 2, 3, ...): Look back N tasks.
            Current {rating} - {rating} from N tasks ago.
        - 'max': Current {rating} - earliest {rating} (first task).
    feature: str
        Must be one of ['quality', 'risk'].

    Returns
    -------
    pd.DataFrame
        Original dataframe with f'{rating}_delta' column added.
    """

    if feature not in ["quality", "risk"]:
        raise ValueError(
            f"Invalid feature: {feature}. Must be one of ['quality', 'risk']."
        )

    # Sort by respondent and date
    df = df.sort_values(["respondentPk", "date_completed"]).copy()

    if delta_period == "max":
        # Compare to earliest quality for each respondent
        df[f"{feature}_delta"] = df[f"{feature}"] - df.groupby("respondentPk")[
            f"{feature}"
        ].transform("first")
    elif isinstance(delta_period, int) and delta_period >= 1:
        # Compare to quality from N periods ago
        df[f"{feature}_delta"] = df.groupby("respondentPk")[f"{feature}"].diff(
            periods=delta_period
        )
    else:
        raise ValueError("delta_period must be a positive integer or 'max'")

    print(f'Rating delta added for {feature} for {delta_period} period(s).')

    return df
