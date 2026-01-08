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
    bins = [0, 10, 25, 75, 125, 175, 225, float("inf")]
    labels = [
        "total_tasks_1_10",
        "total_tasks_11_25",
        "total_tasks_26_75",
        "total_tasks_76_125",
        "total_tasks_126_175",
        "total_tasks_176_225",
        "total_tasks_226_plus",
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