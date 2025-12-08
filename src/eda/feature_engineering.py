"""
Feature Engineering Module

Functions for creating time-based features, behavioral features, and fraud risk scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def create_time_features(df: pd.DataFrame, date_col: str = "date_completed") -> pd.DataFrame:
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
    df["is_night"] = (
        (df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)
    ).astype(int)
    df["is_business_hour"] = (
        (df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 17)
    ).astype(int)
    
    return df


def create_task_speed_features(
    df: pd.DataFrame,
    task_time_col: str = "task_time_taken",
    suspicious_threshold: float = 30.0,
    very_fast_threshold: float = 180.0,
    very_slow_threshold: float = 3600.0
) -> pd.DataFrame:
    """
    Create task speed-related features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing task time data
    task_time_col : str
        Name of the task time column (in seconds)
    suspicious_threshold : float
        Threshold for suspiciously fast tasks (seconds)
    very_fast_threshold : float
        Threshold for very fast tasks (seconds)
    very_slow_threshold : float
        Threshold for very slow tasks (seconds)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added speed features
    """
    df = df.copy()
    df["task_time_minutes"] = df[task_time_col] / 60
    df["is_suspiciously_fast"] = (df[task_time_col] < suspicious_threshold).astype(int)
    df["is_very_fast"] = (df[task_time_col] < very_fast_threshold).astype(int)
    df["is_very_slow"] = (df["task_time_minutes"] > very_slow_threshold / 60).astype(int)
    
    # Points per minute
    # Protect against division by zero (tasks with 0 time would produce inf/nan)
    if "task_points" in df.columns:
        df["task_points_perMinute"] = (
            df["task_points"] / df["task_time_minutes"].replace(0, np.nan)
        )
    
    return df


def create_respondent_behavioral_features(
    df: pd.DataFrame,
    respondent_id_col: str = "respondentPk",
    date_col: str = "date_completed",
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Aggregate task-level data to create respondent-level behavioral features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Task-level DataFrame
    respondent_id_col : str
        Name of the respondent ID column
    date_col : str
        Name of the date column
    config : dict, optional
        Configuration dictionary with thresholds and parameters
        
    Returns:
    --------
    pd.DataFrame
        Respondent-level DataFrame with behavioral features
    """
    if config is None:
        config = {}
    
    df = df.sort_values(by=[respondent_id_col, date_col])
    
    # Aggregation dictionary
    agg_dict = {
        "taskPk": "count",
        "task_time_taken": ["mean", "std", "min", "max", "median"],
        "task_points_perMinute": ["mean", "std", "min", "max", "median"],
        "is_suspiciously_fast": "sum",
        "is_very_fast": "sum",
        "is_night": "sum",
        "is_weekend": "sum",
        "risk": ["mean", "max"],
        "quality": ["mean", "max"],
        "wonky_study_flag": "max",
        date_col: ["min", "max"],
    }
    
    # Filter to only include columns that exist
    filtered_agg_dict = {
        k: v for k, v in agg_dict.items() 
        if k in df.columns
    }
    
    respondent_features = (
        df.groupby(respondent_id_col)
        .agg(filtered_agg_dict)
        .reset_index()
    )
    
    # Flatten column names
    respondent_features.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) and col[1] else col[0]
        for col in respondent_features.columns.values
    ]
    
    # Rename columns
    rename_dict = {
        "taskPk_count": "total_tasks",
        "task_time_taken_mean": "avg_task_time",
        "task_time_taken_std": "std_task_time",
        "task_time_taken_min": "min_task_time",
        "task_time_taken_max": "max_task_time",
        "task_points_perMinute_mean": "avg_task_pointsPerMin",
        "task_points_perMinute_std": "std_task_pointsPerMin",
        "task_points_perMinute_min": "min_task_pointsPerMin",
        "task_points_perMinute_max": "max_task_pointsPerMin",
        "task_time_taken_median": "median_task_time",
        "is_suspiciously_fast_sum": "suspicious_fast_count",
        "is_very_fast_sum": "very_fast_count",
        "is_night_sum": "night_task_count",
        "is_weekend_sum": "weekend_task_count",
        "risk_mean": "avg_risk",
        "risk_max": "max_risk",
        "quality_mean": "avg_quality",
        "quality_max": "max_quality",
        "wonky_study_flag_max": "wonky_study_flag",
        f"{date_col}_min": "first_task_date",
        f"{date_col}_max": "last_task_date",
    }
    
    # Only rename columns that exist
    rename_dict = {k: v for k, v in rename_dict.items() if k in respondent_features.columns}
    respondent_features = respondent_features.rename(columns=rename_dict)
    
    # Calculate rates
    if "total_tasks" in respondent_features.columns:
        if "suspicious_fast_count" in respondent_features.columns:
            respondent_features["suspicious_fast_rate"] = (
                respondent_features["suspicious_fast_count"] / respondent_features["total_tasks"]
            )
        if "very_fast_count" in respondent_features.columns:
            respondent_features["very_fast_rate"] = (
                respondent_features["very_fast_count"] / respondent_features["total_tasks"]
            )
        if "night_task_count" in respondent_features.columns:
            respondent_features["night_task_rate"] = (
                respondent_features["night_task_count"] / respondent_features["total_tasks"]
            )
        if "weekend_task_count" in respondent_features.columns:
            respondent_features["weekend_task_rate"] = (
                respondent_features["weekend_task_count"] / respondent_features["total_tasks"]
            )
    
    # Coefficient of variation
    if "std_task_time" in respondent_features.columns and "avg_task_time" in respondent_features.columns:
        respondent_features["task_time_cv"] = (
            respondent_features["std_task_time"] / 
            respondent_features["avg_task_time"].replace(0, np.nan)
        )
    
    if "std_task_pointsPerMin" in respondent_features.columns and "avg_task_pointsPerMin" in respondent_features.columns:
        respondent_features["task_ppm_cv"] = (
            respondent_features["std_task_pointsPerMin"] / 
            respondent_features["avg_task_pointsPerMin"].replace(0, np.nan)
        )
    
    # Days active and tasks per day
    if "first_task_date" in respondent_features.columns and "last_task_date" in respondent_features.columns:
        respondent_features["days_active"] = (
            respondent_features["last_task_date"] - respondent_features["first_task_date"]
        ).dt.days + 1
        respondent_features["tasks_per_day"] = (
            respondent_features["total_tasks"] / respondent_features["days_active"]
        )
    
    # Volume thresholds
    high_vol_percentile = config.get("high_volume_percentile", 0.95)
    extreme_vol_percentile = config.get("extreme_volume_percentile", 0.99)
    
    if "total_tasks" in respondent_features.columns:
        high_vol_thresh = respondent_features["total_tasks"].quantile(high_vol_percentile)
        extreme_vol_thresh = respondent_features["total_tasks"].quantile(extreme_vol_percentile)
        
        respondent_features["is_high_volume"] = (
            respondent_features["total_tasks"] > high_vol_thresh
        ).astype(int)
        respondent_features["is_extreme_volume"] = (
            respondent_features["total_tasks"] > extreme_vol_thresh
        ).astype(int)
    
    if "avg_task_pointsPerMin" in respondent_features.columns:
        points_perMin_thresh = respondent_features["avg_task_pointsPerMin"].quantile(high_vol_percentile)
        respondent_features["is_high_pointsPerMinute"] = (
            respondent_features["avg_task_pointsPerMin"] > points_perMin_thresh
        ).astype(int)
    
    # Velocity tiers
    velocity_bins = config.get("velocity_bins", [0, 1, 3, 5, 10, np.inf])
    velocity_labels = config.get("velocity_labels", ["Very_Low", "Low", "Medium", "High", "Very_High"])
    
    # Convert string 'inf' from YAML to np.inf for pandas compatibility
    # Also ensure all values are numeric and bins are monotonically increasing
    if isinstance(velocity_bins, list):
        converted_bins = []
        for x in velocity_bins:
            if isinstance(x, str) and x.lower() in ['inf', 'infinity']:
                converted_bins.append(np.inf)
            elif isinstance(x, (int, float)):
                converted_bins.append(float(x))
            else:
                # Try to convert to float, fallback to np.inf if fails
                try:
                    converted_bins.append(float(x))
                except (ValueError, TypeError):
                    converted_bins.append(np.inf)
        velocity_bins = converted_bins
    
    # Ensure bins are monotonically increasing (safety check)
    # Convert to list of floats, remove duplicates while preserving order, then sort
    seen = set()
    unique_bins = []
    for x in velocity_bins:
        if x not in seen:
            seen.add(x)
            unique_bins.append(x)
    velocity_bins = sorted(unique_bins)  # Sort to ensure monotonic increase
    
    if "tasks_per_day" in respondent_features.columns:
        try:
            respondent_features["velocity_tier"] = pd.cut(
                respondent_features["tasks_per_day"],
                bins=velocity_bins,
                labels=velocity_labels,
            )
        except ValueError as e:
            # If bins still fail, use default bins
            print(f"Warning: velocity_bins conversion failed, using default bins. Error: {e}")
            velocity_bins = [0, 1, 3, 5, 10, np.inf]
            respondent_features["velocity_tier"] = pd.cut(
                respondent_features["tasks_per_day"],
                bins=velocity_bins,
                labels=velocity_labels,
            )
    
    if "avg_task_pointsPerMin" in respondent_features.columns:
        try:
            respondent_features["velocity_tier(pointsPerMinute)"] = pd.cut(
                respondent_features["avg_task_pointsPerMin"],
                bins=velocity_bins,
                labels=velocity_labels,
            )
        except ValueError as e:
            # If bins still fail, use default bins
            print(f"Warning: velocity_bins conversion failed for pointsPerMinute, using default bins. Error: {e}")
            velocity_bins = [0, 1, 3, 5, 10, np.inf]
            respondent_features["velocity_tier(pointsPerMinute)"] = pd.cut(
                respondent_features["avg_task_pointsPerMin"],
                bins=velocity_bins,
                labels=velocity_labels,
            )
    
    return respondent_features


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
        DataFrame with wonky study information
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
    
    # Fill NaN with 0
    for col in available_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Wonky concentration
    if "wonky_task_ratio" in df.columns:
        df['wonky_concentration'] = pd.cut(
            df['wonky_task_ratio'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
            include_lowest=True
        )
        df['is_high_wonky'] = (df['wonky_task_ratio'] > 0.5).astype(int)
        df['is_quite_wonky'] = (df['wonky_task_ratio'] > 0.3).astype(int)
        
        if "wonky_unique_tasks" in df.columns and "wonky_task_instances" in df.columns:
            df['wonky_diversity'] = (
                df['wonky_unique_tasks'] / 
                df['wonky_task_instances'].replace(0, np.nan)
            )
    
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

