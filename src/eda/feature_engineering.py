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


def create_task_speed_features(
    df: pd.DataFrame,
    task_time_col: str = "task_time_taken_s",
    use_std_dev: bool = True
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
        
        # Filter out NaN and invalid values for percentile calculation
        valid_times = df[task_time_col].dropna()
        if len(valid_times) == 0:
            raise ValueError(f"No valid values found in '{task_time_col}' column")
        
        # Check if all values are the same (would make percentiles meaningless)
        if valid_times.nunique() == 1:
            # If all values are the same, set all flags appropriately
            df["is_fast"] = 0
            df["is_suspiciously_fast"] = 0
            df["is_slow"] = 0
            df["is_suspiciously_slow"] = 0
            df["is_normal_speed"] = 1  # All tasks are "normal" if they're all the same
            df["is_very_fast"] = 0
        else:
            # Calculate percentiles for the entire sample
            fast_threshold = valid_times.quantile(0.16)  # ~1 std dev below mean
            suspiciously_fast_threshold = valid_times.quantile(0.025)  # ~2 std dev below mean
            slow_threshold = valid_times.quantile(0.84)  # ~1 std dev above mean
            suspiciously_slow_threshold = valid_times.quantile(0.975)  # ~2 std dev above mean
            
            # Also calculate mean and std for display (using trimmed data to avoid extreme outliers)
            # Trim extreme outliers (top/bottom 1%) before calculating stats for display
            trimmed_data = valid_times.clip(
                lower=valid_times.quantile(0.01),
                upper=valid_times.quantile(0.99)
            )
            mean_time = trimmed_data.mean()
            std_time = trimmed_data.std()
            
            # Create speed flags (handle NaN values by setting to 0)
            df["is_fast"] = (df[task_time_col] < fast_threshold).fillna(0).astype(int)
            df["is_suspiciously_fast"] = (df[task_time_col] < suspiciously_fast_threshold).fillna(0).astype(int)
            df["is_slow"] = (df[task_time_col] > slow_threshold).fillna(0).astype(int)
            df["is_suspiciously_slow"] = (df[task_time_col] > suspiciously_slow_threshold).fillna(0).astype(int)
            
            # Normal speed: within 1 std dev of mean (between 16th and 84th percentile)
            df["is_normal_speed"] = (
                (df[task_time_col] >= fast_threshold) & 
                (df[task_time_col] <= slow_threshold)
            ).fillna(0).astype(int)
            
            # Store thresholds for reference (as metadata in a comment or as attributes)
            # For backward compatibility, also create legacy column names
            df["is_very_fast"] = df["is_fast"].copy()
        
        # Store calculated thresholds as attributes for potential future reference
        # (Note: DataFrame attributes aren't preserved, but we'll use them in the notebook)
    else:
        # Legacy hardcoded thresholds (for backward compatibility)
        suspicious_threshold = 30.0
        very_fast_threshold = 180.0
        very_slow_threshold = 3600.0
        
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
    categorical_cols: list = None,
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
    categorical_cols : list
        List of categorical column names to include in grouping and preserve in output.
        Examples: ['platform_name', 'hardware_version', 'survey_locale']
    config : dict, optional
        Configuration dictionary with thresholds and parameters
        
    Returns:
    --------
    pd.DataFrame
        Respondent-level DataFrame with behavioral features and demographic columns
    """
    if config is None:
        config = {}
    
    # Handle None demographic_cols
    if categorical_cols is None:
        categorical_cols = []
    
    # Filter demographic columns to only those that exist in the dataframe
    available_categorical_cols = [
        col for col in categorical_cols 
        if col in df.columns and col != respondent_id_col
    ]
    
    # Sort by respondent ID, demographics (if any), and date
    sort_cols = [respondent_id_col] + available_categorical_cols + [date_col]
    df = df.sort_values(by=sort_cols)

    print(df.shape)
    
    # Aggregation dictionary for behavioral features
    agg_dict = {
        "taskPk": "count",
        "task_time_taken_s": ["mean", "std", "min", "max", "median"],
        "task_time_taken_s_adj": ["mean", "std", "min", "max", "median"],
        "task_points_perMinute": ["mean", "std", "min", "max", "median"],
        "is_suspiciously_fast": "sum",
        "is_very_fast": "sum",
        "is_fast": "sum",  # New: 1 std dev below mean
        "is_slow": "sum",  # New: 1 std dev above mean
        "is_suspiciously_slow": "sum",  # New: 2 std dev above mean
        "is_normal_speed": "sum",  # New: within 1 std dev of mean (16th-84th percentile)
        "is_night": "sum",
        "is_weekend": "sum",
        "risk": ["mean", "max"],
        "quality": ["mean", "max"],
        "wonky_study_flag": "max",
        "wonky_study_count": "sum",
        "days_active_before_task": "mean",
        date_col: ["min", "max"],
    }
    
    # Filter to only include columns that exist
    filtered_agg_dict = {
        k: v for k, v in agg_dict.items() 
        if k in df.columns
    }
    
    # Group by ONLY respondent ID (matching original behavior)
    # This ensures one row per respondent regardless of demographic variations
    respondent_features = (
        df.groupby(respondent_id_col)
        .agg(filtered_agg_dict)
        .reset_index()
    )

    print(respondent_features.shape)
    
    # Flatten column names
    # Aggregated columns are tuples like ('taskPk', 'count')
    flattened_cols = []
    for col in respondent_features.columns.values:
        if isinstance(col, tuple):
            # Aggregated column: join tuple elements
            flattened_cols.append("_".join(str(c) for c in col).strip("_"))
        else:
            # Grouping column (respondent_id_col): keep as string
            flattened_cols.append(str(col))
    respondent_features.columns = flattened_cols
    
    # Add demographic columns by taking the mode (most common value) per respondent
    # If mode is not available (all NaN), use first value
    if available_categorical_cols:
        def get_mode_or_first(series):
            """Get mode if available, otherwise first non-null value, otherwise first value."""
            # Remove NaN values for mode calculation
            non_null = series.dropna()
            if len(non_null) > 0:
                mode_values = non_null.mode()
                if len(mode_values) > 0:
                    return mode_values.iloc[0]
                else:
                    return non_null.iloc[0]
            else:
                # All NaN, return first value (which is NaN)
                return series.iloc[0] if len(series) > 0 else None
        
        categorical_agg = {}
        for col in available_categorical_cols:
            if col in df.columns:
                categorical_agg[col] = get_mode_or_first
        
        if categorical_agg:
            categorical_features = (
                df.groupby(respondent_id_col)
                .agg(categorical_agg)
                .reset_index()
            )
            # Merge demographic columns into respondent_features
            respondent_features = respondent_features.merge(
                categorical_features,
                on=respondent_id_col,
                how='left'
            )
    
    # Rename columns
    rename_dict = {
        "taskPk_count": "total_tasks",
        "task_time_taken_s_mean": "avg_task_time",
        "task_time_taken_s_std": "std_task_time",
        "task_time_taken_s_min": "min_task_time",
        "task_time_taken_s_max": "max_task_time",
        "task_time_taken_s_median": "median_task_time",
        "task_time_taken_s_adj_mean": "avg_task_time_adj",
        "task_time_taken_s_adj_std": "std_task_time_adj",
        "task_time_taken_s_adj_min": "min_task_time_adj",
        "task_time_taken_s_adj_max": "max_task_time_adj",
        "task_time_taken_s_adj_median": "median_task_time_adj",
        "task_points_perMinute_mean": "avg_task_pointsPerMin",
        "task_points_perMinute_std": "std_task_pointsPerMin",
        "task_points_perMinute_min": "min_task_pointsPerMin",
        "task_points_perMinute_max": "max_task_pointsPerMin",
        "task_points_perMinute_median": "median_task_pointsPerMin",
        "is_suspiciously_fast_sum": "suspicious_fast_count",
        "is_very_fast_sum": "very_fast_count",
        "is_fast_sum": "fast_count",
        "is_slow_sum": "slow_count",
        "is_suspiciously_slow_sum": "suspiciously_slow_count",
        "is_normal_speed_sum": "normal_speed_count",
        "is_night_sum": "night_task_count",
        "is_weekend_sum": "weekend_task_count",
        "risk_mean": "avg_risk",
        "risk_max": "max_risk",
        "quality_mean": "avg_quality",
        "quality_max": "max_quality",
        "wonky_study_flag_max": "wonky_study_flag",
        "wonky_study_count_sum": "wonky_study_count",
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
        if "fast_count" in respondent_features.columns:
            respondent_features["fast_rate"] = (
                respondent_features["fast_count"] / respondent_features["total_tasks"]
            )
        if "slow_count" in respondent_features.columns:
            respondent_features["slow_rate"] = (
                respondent_features["slow_count"] / respondent_features["total_tasks"]
            )
        if "suspiciously_slow_count" in respondent_features.columns:
            respondent_features["suspiciously_slow_rate"] = (
                respondent_features["suspiciously_slow_count"] / respondent_features["total_tasks"]
            )
        if "normal_speed_count" in respondent_features.columns:
            respondent_features["normal_speed_rate"] = (
                respondent_features["normal_speed_count"] / respondent_features["total_tasks"]
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
        respondent_features["tasks_per_days_active_all"] = (
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
    
    if "tasks_per_days_active_all" in respondent_features.columns:
        try:
            respondent_features["velocity_tier"] = pd.cut(
                respondent_features["tasks_per_days_active_all"],
                bins=velocity_bins,
                labels=velocity_labels,
            )
        except:
            pass
    
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
    
    # Fill NaN with 0
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