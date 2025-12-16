"""
Diagnostic utilities for investigating task_time_taken_s calculation issues.

This module provides functions to analyze and validate task time calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def analyze_task_time_distribution(
    df: pd.DataFrame,
    task_time_col: str = "task_time_taken_s"
) -> Dict:
    """
    Analyze the distribution of task_time_taken_s values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing task_time_taken_s column
    task_time_col : str
        Name of the task time column
        
    Returns:
    --------
    dict
        Dictionary containing distribution statistics
    """
    if task_time_col not in df.columns:
        return {"error": f"Column '{task_time_col}' not found in DataFrame"}
    
    series = df[task_time_col]
    
    stats = {
        "count": len(series),
        "null_count": series.isnull().sum(),
        "null_percentage": (series.isnull().sum() / len(series)) * 100,
        "non_null_count": series.notna().sum(),
    }
    
    if stats["non_null_count"] > 0:
        valid_series = series.dropna()
        
        stats.update({
            "mean": float(valid_series.mean()),
            "median": float(valid_series.median()),
            "std": float(valid_series.std()),
            "min": float(valid_series.min()),
            "max": float(valid_series.max()),
            "percentiles": {
                "p1": float(valid_series.quantile(0.01)),
                "p5": float(valid_series.quantile(0.05)),
                "p25": float(valid_series.quantile(0.25)),
                "p50": float(valid_series.quantile(0.50)),
                "p75": float(valid_series.quantile(0.75)),
                "p95": float(valid_series.quantile(0.95)),
                "p99": float(valid_series.quantile(0.99)),
            },
            "negative_count": int((valid_series < 0).sum()),
            "zero_count": int((valid_series == 0).sum()),
            "outliers": {
                "greater_than_1_hour": int((valid_series > 3600).sum()),
                "greater_than_1_day": int((valid_series > 86400).sum()),
                "greater_than_1_week": int((valid_series > 604800).sum()),
            }
        })
        
        # Convert to minutes for readability
        stats["mean_minutes"] = stats["mean"] / 60
        stats["median_minutes"] = stats["median"] / 60
        stats["std_minutes"] = stats["std"] / 60
        stats["min_minutes"] = stats["min"] / 60
        stats["max_minutes"] = stats["max"] / 60
    
    return stats


def validate_date_columns(
    df: pd.DataFrame,
    date_completed_col: str = "date_completed",
    date_created_col: str = "date_created"
) -> Dict:
    """
    Validate date columns used in task_time_taken_s calculation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing date columns
    date_completed_col : str
        Name of date completed column
    date_created_col : str
        Name of date created column
        
    Returns:
    --------
    dict
        Dictionary containing validation results
    """
    validation = {
        "date_completed": {
            "exists": date_completed_col in df.columns,
            "null_count": 0,
            "null_percentage": 0.0,
            "dtype": None,
            "min_date": None,
            "max_date": None,
            "sample_values": []
        },
        "date_created": {
            "exists": date_created_col in df.columns,
            "null_count": 0,
            "null_percentage": 0.0,
            "dtype": None,
            "min_date": None,
            "max_date": None,
            "sample_values": []
        }
    }
    
    # Check date_completed
    if validation["date_completed"]["exists"]:
        col_data = df[date_completed_col]
        validation["date_completed"]["null_count"] = int(col_data.isnull().sum())
        validation["date_completed"]["null_percentage"] = (
            validation["date_completed"]["null_count"] / len(col_data) * 100
        )
        validation["date_completed"]["dtype"] = str(col_data.dtype)
        
        if col_data.notna().sum() > 0:
            valid_data = col_data.dropna()
            try:
                # Try to convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(valid_data):
                    valid_data = pd.to_datetime(valid_data, errors='coerce')
                
                validation["date_completed"]["min_date"] = str(valid_data.min())
                validation["date_completed"]["max_date"] = str(valid_data.max())
                validation["date_completed"]["sample_values"] = [
                    str(x) for x in valid_data.head(5).tolist()
                ]
            except Exception as e:
                validation["date_completed"]["error"] = str(e)
    
    # Check date_created
    if validation["date_created"]["exists"]:
        col_data = df[date_created_col]
        validation["date_created"]["null_count"] = int(col_data.isnull().sum())
        validation["date_created"]["null_percentage"] = (
            validation["date_created"]["null_count"] / len(col_data) * 100
        )
        validation["date_created"]["dtype"] = str(col_data.dtype)
        
        if col_data.notna().sum() > 0:
            valid_data = col_data.dropna()
            try:
                # Try to convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(valid_data):
                    valid_data = pd.to_datetime(valid_data, errors='coerce')
                
                validation["date_created"]["min_date"] = str(valid_data.min())
                validation["date_created"]["max_date"] = str(valid_data.max())
                validation["date_created"]["sample_values"] = [
                    str(x) for x in valid_data.head(5).tolist()
                ]
            except Exception as e:
                validation["date_created"]["error"] = str(e)
    
    # Check logical consistency if both columns exist
    if (validation["date_completed"]["exists"] and 
        validation["date_created"]["exists"]):
        
        both_valid = (
            df[date_completed_col].notna() & 
            df[date_created_col].notna()
        )
        
        if both_valid.sum() > 0:
            try:
                completed = pd.to_datetime(df.loc[both_valid, date_completed_col], errors='coerce')
                created = pd.to_datetime(df.loc[both_valid, date_created_col], errors='coerce')
                
                time_diff = (completed - created).dt.total_seconds()
                
                validation["logical_checks"] = {
                    "both_valid_count": int(both_valid.sum()),
                    "negative_time_diff_count": int((time_diff < 0).sum()),
                    "unreasonably_large_count": int((time_diff > 86400).sum()),  # > 1 day
                    "mean_time_diff_seconds": float(time_diff.mean()) if len(time_diff) > 0 else None,
                    "median_time_diff_seconds": float(time_diff.median()) if len(time_diff) > 0 else None,
                }
            except Exception as e:
                validation["logical_checks"] = {"error": str(e)}
    
    return validation


def print_diagnostic_report(
    df: pd.DataFrame,
    task_time_col: str = "task_time_taken_s",
    date_completed_col: str = "date_completed",
    date_created_col: str = "date_created"
) -> None:
    """
    Print a comprehensive diagnostic report for task_time_taken_s.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    task_time_col : str
        Name of task time column
    date_completed_col : str
        Name of date completed column
    date_created_col : str
        Name of date created column
    """
    print("=" * 80)
    print("TASK_TIME_TAKEN_S DIAGNOSTIC REPORT")
    print("=" * 80)
    print()
    
    # Analyze task time distribution
    print("1. TASK TIME DISTRIBUTION ANALYSIS")
    print("-" * 80)
    time_stats = analyze_task_time_distribution(df, task_time_col)
    
    if "error" in time_stats:
        print(f"ERROR: {time_stats['error']}")
    else:
        print(f"Total records: {time_stats['count']:,}")
        print(f"NULL values: {time_stats['null_count']:,} ({time_stats['null_percentage']:.2f}%)")
        print(f"Valid values: {time_stats['non_null_count']:,}")
        print()
        
        if time_stats['non_null_count'] > 0:
            print("Statistics (seconds):")
            print(f"  Mean: {time_stats['mean']:.2f} ({time_stats['mean_minutes']:.2f} minutes)")
            print(f"  Median: {time_stats['median']:.2f} ({time_stats['median_minutes']:.2f} minutes)")
            print(f"  Std Dev: {time_stats['std']:.2f} ({time_stats['std_minutes']:.2f} minutes)")
            print(f"  Min: {time_stats['min']:.2f} ({time_stats['min_minutes']:.2f} minutes)")
            print(f"  Max: {time_stats['max']:.2f} ({time_stats['max_minutes']:.2f} minutes)")
            print()
            
            print("Percentiles (seconds):")
            for p_name, p_value in time_stats['percentiles'].items():
                print(f"  {p_name}: {p_value:.2f} ({p_value/60:.2f} minutes)")
            print()
            
            print("Data Quality Issues:")
            print(f"  Negative values: {time_stats['negative_count']:,}")
            print(f"  Zero values: {time_stats['zero_count']:,}")
            print(f"  > 1 hour: {time_stats['outliers']['greater_than_1_hour']:,}")
            print(f"  > 1 day: {time_stats['outliers']['greater_than_1_day']:,}")
            print(f"  > 1 week: {time_stats['outliers']['greater_than_1_week']:,}")
    
    print()
    print("=" * 80)
    print("2. DATE COLUMNS VALIDATION")
    print("-" * 80)
    
    date_validation = validate_date_columns(df, date_completed_col, date_created_col)
    
    # Date completed
    print(f"\ndate_completed column ('{date_completed_col}'):")
    if date_validation["date_completed"]["exists"]:
        print(f"  ✓ Column exists")
        print(f"  Data type: {date_validation['date_completed']['dtype']}")
        print(f"  NULL count: {date_validation['date_completed']['null_count']:,} "
              f"({date_validation['date_completed']['null_percentage']:.2f}%)")
        if date_validation["date_completed"]["min_date"]:
            print(f"  Min date: {date_validation['date_completed']['min_date']}")
            print(f"  Max date: {date_validation['date_completed']['max_date']}")
            print(f"  Sample values: {date_validation['date_completed']['sample_values'][:3]}")
        if "error" in date_validation["date_completed"]:
            print(f"  ERROR: {date_validation['date_completed']['error']}")
    else:
        print(f"  ✗ Column does NOT exist")
    
    # Date created
    print(f"\ndate_created column ('{date_created_col}'):")
    if date_validation["date_created"]["exists"]:
        print(f"  ✓ Column exists")
        print(f"  Data type: {date_validation['date_created']['dtype']}")
        print(f"  NULL count: {date_validation['date_created']['null_count']:,} "
              f"({date_validation['date_created']['null_percentage']:.2f}%)")
        if date_validation["date_created"]["min_date"]:
            print(f"  Min date: {date_validation['date_created']['min_date']}")
            print(f"  Max date: {date_validation['date_created']['max_date']}")
            print(f"  Sample values: {date_validation['date_created']['sample_values'][:3]}")
        if "error" in date_validation["date_created"]:
            print(f"  ERROR: {date_validation['date_created']['error']}")
    else:
        print(f"  ✗ Column does NOT exist")
        print(f"  WARNING: This column is required for task_time_taken_s calculation!")
    
    # Logical checks
    if "logical_checks" in date_validation:
        print(f"\nLogical Consistency Checks:")
        checks = date_validation["logical_checks"]
        if "error" in checks:
            print(f"  ERROR: {checks['error']}")
        else:
            print(f"  Records with both dates valid: {checks['both_valid_count']:,}")
            print(f"  Negative time differences (created > completed): {checks['negative_time_diff_count']:,}")
            print(f"  Unreasonably large differences (> 1 day): {checks['unreasonably_large_count']:,}")
            if checks['mean_time_diff_seconds']:
                print(f"  Mean time difference: {checks['mean_time_diff_seconds']:.2f}s "
                      f"({checks['mean_time_diff_seconds']/60:.2f} minutes)")
                print(f"  Median time difference: {checks['median_time_diff_seconds']:.2f}s "
                      f"({checks['median_time_diff_seconds']/60:.2f} minutes)")
    
    print()
    print("=" * 80)
    print("DIAGNOSTIC REPORT COMPLETE")
    print("=" * 80)

