"""
Validators Module

Data quality validation functions.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict


def check_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Check for duplicate rows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    subset : List[str], optional
        Columns to check for duplicates
        
    Returns:
    --------
    Dict[str, int]
        Dictionary with duplicate count
    """
    if subset:
        duplicates = df.duplicated(subset=subset).sum()
    else:
        duplicates = df.duplicated().sum()
    
    return {'duplicate_count': duplicates}


def check_null_percentage(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_percentage: float = 0.5
) -> Dict[str, Dict]:
    """
    Check null percentage for columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    columns : List[str], optional
        Columns to check (default: all columns)
    max_percentage : float
        Maximum acceptable null percentage
        
    Returns:
    --------
    Dict[str, Dict]
        Dictionary with null statistics
    """
    if columns is None:
        columns = df.columns.tolist()
    
    null_stats = {}
    high_null_cols = []
    
    for col in columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = null_count / len(df)
            null_stats[col] = {
                'null_count': int(null_count),
                'null_percentage': float(null_pct)
            }
            
            if null_pct > max_percentage:
                high_null_cols.append(col)
    
    return {
        'null_statistics': null_stats,
        'high_null_columns': high_null_cols,
        'max_percentage': max_percentage
    }


def check_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 3.0
) -> Dict[str, Dict]:
    """
    Check for outliers using IQR or Z-score method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    columns : List[str], optional
        Columns to check (default: numeric columns)
    method : str
        Method to use ("iqr" or "zscore")
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    Dict[str, Dict]
        Dictionary with outlier statistics
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_stats = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        
        if method == "iqr":
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        elif method == "zscore":
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = (z_scores > threshold).sum()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_stats[col] = {
            'outlier_count': int(outliers),
            'outlier_percentage': float(outliers / len(col_data)) if len(col_data) > 0 else 0.0
        }
    
    return outlier_stats


def validate_data_quality(
    df: pd.DataFrame,
    duplicate_subset: Optional[List[str]] = None,
    check_nulls: bool = True,
    check_outliers: bool = True,
    max_null_percentage: float = 0.5
) -> Dict[str, Dict]:
    """
    Comprehensive data quality validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    duplicate_subset : List[str], optional
        Columns to check for duplicates
    check_nulls : bool
        Whether to check null percentages
    check_outliers : bool
        Whether to check outliers
    max_null_percentage : float
        Maximum acceptable null percentage
        
    Returns:
    --------
    Dict[str, Dict]
        Dictionary with all validation results
    """
    results = {}
    
    # Check duplicates
    results['duplicates'] = check_duplicates(df, duplicate_subset)
    
    # Check nulls
    if check_nulls:
        results['nulls'] = check_null_percentage(df, max_percentage=max_null_percentage)
    
    # Check outliers
    if check_outliers:
        results['outliers'] = check_outliers(df)
    
    return results

