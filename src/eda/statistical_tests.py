"""
Statistical Tests Module

Functions for performing statistical hypothesis tests and analysis.
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency
from typing import List, Dict, Optional, Tuple


def perform_mannwhitney_test(
    group1: pd.Series,
    group2: pd.Series,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test between two groups.
    
    Parameters:
    -----------
    group1 : pd.Series
        First group data
    group2 : pd.Series
        Second group data
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', or 'greater'
        
    Returns:
    --------
    tuple
        (statistic, p_value)
    """
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()
    
    if len(group1_clean) == 0 or len(group2_clean) == 0:
        return np.nan, np.nan
    
    statistic, p_value = mannwhitneyu(
        group1_clean,
        group2_clean,
        alternative=alternative
    )
    
    return statistic, p_value


def perform_welch_ttest(
    group1: pd.Series,
    group2: pd.Series,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform Welch's t-test (unequal variances t-test) between two groups.
    
    This is a parametric test that doesn't assume equal variances, making it
    more robust than the standard t-test. Useful as a validator/sense check
    alongside non-parametric tests like Mann-Whitney U.
    
    Parameters:
    -----------
    group1 : pd.Series
        First group data
    group2 : pd.Series
        Second group data
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', or 'greater'
        Note: scipy's ttest_ind only supports 'two-sided', but we include
        this parameter for consistency with other test functions
        
    Returns:
    --------
    tuple
        (statistic, p_value)
    """
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()
    
    if len(group1_clean) == 0 or len(group2_clean) == 0:
        return np.nan, np.nan
    
    # Check if we have enough data (need at least 2 observations per group)
    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan, np.nan
    
    # Perform Welch's t-test (equal_var=False means Welch's test)
    # Note: alternative parameter available in scipy >= 1.7.0
    try:
        statistic, p_value = ttest_ind(
            group1_clean,
            group2_clean,
            equal_var=False,  # This makes it Welch's t-test
            alternative=alternative
        )
    except TypeError:
        # Fallback for older scipy versions that don't support alternative parameter
        statistic, p_value = ttest_ind(
            group1_clean,
            group2_clean,
            equal_var=False  # This makes it Welch's t-test
        )
        # For one-sided tests in older scipy, we'd need to manually adjust p-value
        # but for now, we'll just use two-sided
    
    return statistic, p_value


def compare_groups_statistically(
    df: pd.DataFrame,
    group_col: str,
    metrics: List[str],
    group1_value: int = 1,
    group2_value: int = 0,
    significance_level: float = 0.05,
    include_welch: bool = True
) -> pd.DataFrame:
    """
    Compare two groups across multiple metrics using Mann-Whitney U test
    and optionally Welch's t-test as a validator.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with group labels and metrics
    group_col : str
        Name of the column containing group labels
    metrics : List[str]
        List of metric column names to test
    group1_value : int
        Value for first group (e.g., 1 for wonky users)
    group2_value : int
        Value for second group (e.g., 0 for non-wonky users)
    significance_level : float
        Significance level for determining significance
    include_welch : bool
        Whether to include Welch's t-test results as a validator
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with test results for each metric
    """
    group1 = df[df[group_col] == group1_value]
    group2 = df[df[group_col] == group2_value]
    
    results = []
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        group1_values = group1[metric].dropna()
        group2_values = group2[metric].dropna()
        
        if len(group1_values) == 0 or len(group2_values) == 0:
            continue
        
        # Perform Mann-Whitney U test (primary test)
        mw_statistic, mw_p_value = perform_mannwhitney_test(
            group1_values,
            group2_values,
            alternative='two-sided'
        )
        
        # Perform Welch's t-test as validator (if requested)
        welch_statistic = np.nan
        welch_p_value = np.nan
        tests_agree = np.nan
        
        if include_welch:
            welch_statistic, welch_p_value = perform_welch_ttest(
                group1_values,
                group2_values,
                alternative='two-sided'
            )
            
            # Check if both tests agree on significance
            if not (np.isnan(mw_p_value) or np.isnan(welch_p_value)):
                mw_sig = mw_p_value < significance_level
                welch_sig = welch_p_value < significance_level
                tests_agree = mw_sig == welch_sig
        
        # Calculate descriptive statistics
        group1_median = group1_values.median()
        group2_median = group2_values.median()
        median_diff = group1_median - group2_median
        
        group1_mean = group1_values.mean()
        group2_mean = group2_values.mean()
        mean_diff = group1_mean - group2_mean
        
        # Calculate standard deviations
        group1_std = group1_values.std()
        group2_std = group2_values.std()
        
        result_dict = {
            'metric': metric,
            'wonky_median': group1_median,
            'non_wonky_median': group2_median,
            'median_difference': median_diff,
            'wonky_mean': group1_mean,
            'non_wonky_mean': group2_mean,
            'mean_difference': mean_diff,
            'wonky_std': group1_std,
            'non_wonky_std': group2_std,
            'mw_statistic': mw_statistic,
            'mw_p_value': mw_p_value,
            'mw_significant': mw_p_value < significance_level if not np.isnan(mw_p_value) else False,
        }
        
        if include_welch:
            result_dict.update({
                'welch_statistic': welch_statistic,
                'welch_p_value': welch_p_value,
                'welch_significant': welch_p_value < significance_level if not np.isnan(welch_p_value) else False,
                'tests_agree': tests_agree
            })
        
        results.append(result_dict)
    
    results_df = pd.DataFrame(results)
    
    # Sort by Mann-Whitney p-value (primary test)
    if 'mw_p_value' in results_df.columns:
        results_df = results_df.sort_values('mw_p_value')
    
    return results_df


def analyze_thresholds(
    df: pd.DataFrame,
    feature: str,
    bins: int = 10,
    target_col: Optional[str] = None,
    agg_funcs: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze target rates across feature bins using quantile-based binning.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with feature and target columns
    feature : str
        Name of the feature column to bin
    bins : int
        Number of quantile bins to create
    target_col : str, optional
        Name of the target column (e.g., 'has_wonky_tasks')
        If None, will try to use 'has_wonky_tasks' or 'wonky_task_ratio'
    agg_funcs : List[str], optional
        List of aggregation functions to apply
        Default: ['sum', 'count', 'mean'] for binary target,
                 ['mean'] for continuous target
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with threshold analysis results
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame")
    
    df_clean = df[[feature]].dropna()
    
    if len(df_clean) == 0:
        return pd.DataFrame()
    
    # Determine target column
    if target_col is None:
        if 'has_wonky_tasks' in df.columns:
            target_col = 'has_wonky_tasks'
        elif 'wonky_task_ratio' in df.columns:
            target_col = 'wonky_task_ratio'
        else:
            raise ValueError("No target column specified and no default found")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Create bins
    df_analysis = df[[feature, target_col]].dropna()
    
    if len(df_analysis) == 0:
        return pd.DataFrame()
    
    df_analysis[f'{feature}_bin'] = pd.qcut(
        df_analysis[feature].rank(method='first'),
        q=bins,
        duplicates='drop'
    )
    
    # Determine aggregation functions
    if agg_funcs is None:
        if df[target_col].dtype in ['int64', 'int32', 'bool']:
            agg_funcs = ['sum', 'count', 'mean']
        else:
            agg_funcs = ['mean']
    
    # Aggregate
    agg_dict = {
        target_col: agg_funcs,
        feature: ['min', 'max', 'mean']
    }
    
    threshold_analysis = df_analysis.groupby(f'{feature}_bin').agg(agg_dict).reset_index()
    
    # Flatten column names
    threshold_analysis.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) and col[1] else col[0]
        for col in threshold_analysis.columns.values
    ]
    
    # Rename columns for clarity
    if 'has_wonky_tasks' in target_col or 'wonky_task_ratio' in target_col:
        rename_dict = {
            f'{target_col}_sum': 'wonky_count',
            f'{target_col}_count': 'total_count',
            f'{target_col}_mean': 'wonky_rate',
            f'{feature}_min': 'min_val',
            f'{feature}_max': 'max_val',
            f'{feature}_mean': 'mean_val'
        }
    else:
        rename_dict = {
            f'{target_col}_mean': 'target_mean',
            f'{feature}_min': 'min_val',
            f'{feature}_max': 'max_val',
            f'{feature}_mean': 'mean_val'
        }
    
    # Only rename columns that exist
    rename_dict = {k: v for k, v in rename_dict.items() if k in threshold_analysis.columns}
    threshold_analysis = threshold_analysis.rename(columns=rename_dict)
    
    return threshold_analysis


def compare_groups_with_both_tests(
    df: pd.DataFrame,
    group_col: str,
    metrics: List[str],
    group1_value: int = 1,
    group2_value: int = 0,
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    Compare two groups using both Mann-Whitney U and Welch's t-test.
    This function explicitly runs both tests for comparison and validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with group labels and metrics
    group_col : str
        Name of the column containing group labels
    metrics : List[str]
        List of metric column names to test
    group1_value : int
        Value for first group (e.g., 1 for wonky users)
    group2_value : int
        Value for second group (e.g., 0 for non-wonky users)
    significance_level : float
        Significance level for determining significance
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with results from both tests for each metric
    """
    return compare_groups_statistically(
        df=df,
        group_col=group_col,
        metrics=metrics,
        group1_value=group1_value,
        group2_value=group2_value,
        significance_level=significance_level,
        include_welch=True
    )


def format_hypothesis_results(
    statistical_results: pd.DataFrame,
    hypotheses: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    """
    Format statistical test results with hypothesis information.
    
    Parameters:
    -----------
    statistical_results : pd.DataFrame
        DataFrame from compare_groups_statistically
    hypotheses : dict, optional
        Dictionary mapping metric names to hypothesis descriptions
        
    Returns:
    --------
    pd.DataFrame
        Formatted results with hypothesis information
    """
    results = statistical_results.copy()
    
    if hypotheses:
        results['hypothesis'] = results['metric'].map(
            lambda x: hypotheses.get(x, {}).get('description', '')
        )
        results['expected_direction'] = results['metric'].map(
            lambda x: hypotheses.get(x, {}).get('expected_direction', '')
        )
    
    return results


def perform_chi_square_tests(
    df: pd.DataFrame,
    feature_set: List[str],
    group_var: str = "wonky_study_count",
    significance_level: float = 0.01
) -> pd.DataFrame:
    """
    Perform chi-squared tests for independence between temporal features and group variable.
    
    Tests whether temporal features are independent of the group variable (wonky vs non-wonky).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and group variable
    feature_set : List[str]
        List of temporal feature column names to test
    group_var : str
        Column name for grouping variable
    significance_level : float
        Significance level for determining significance
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with chi2, p_value, and significance for each feature
    """
    results = []
    
    # Prepare group variable (binary: >0 vs =0 or NaN)
    if group_var in df.columns:
        df_test = df.copy()
        df_test[group_var] = df_test[group_var].fillna(0)
        df_test['group_binary'] = (df_test[group_var] > 0).astype(int)
    else:
        return pd.DataFrame()
    
    for feature in feature_set:
        if feature not in df_test.columns:
            continue
        
        # Create contingency table
        contingency_table = pd.crosstab(
            df_test['group_binary'],
            df_test[feature]
        )
        
        # Check if we have enough data
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            continue
        
        # Perform chi-squared test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            results.append({
                'feature': feature,
                'chi2': chi2,
                'chi_p_value': p_value,
                'dof': dof,
                'significant': p_value < significance_level
            })
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df = results_df.set_index('feature')
        results_df = results_df.sort_values('chi2', ascending=False)
    
    return results_df


def compare_demographic_groups(
    df: pd.DataFrame,
    demographic_col: str,
    target_col: str = "wonky_task_ratio",
    min_group_size: int = 10,
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    Compare wonky task rates across demographic groups using Mann-Whitney U and Welch's t-test.
    
    For each pair of demographic groups, performs statistical tests to identify
    significant differences in wonky task rates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with demographic and target columns
    demographic_col : str
        Column name for demographic grouping (e.g., 'platform_name', 'hardware_version', 'survey_locale')
    target_col : str
        Column name for target metric (e.g., 'wonky_task_ratio' or 'has_wonky_tasks')
    min_group_size : int
        Minimum number of observations required per group for testing
    significance_level : float
        Significance level for determining significance
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with comparison results for each group pair
    """
    if demographic_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()
    
    # Get unique groups
    df_clean = df[[demographic_col, target_col]].dropna()
    
    if len(df_clean) == 0:
        return pd.DataFrame()
    
    groups = df_clean[demographic_col].unique()
    
    if len(groups) < 2:
        return pd.DataFrame()
    
    results = []
    
    # Compare each pair of groups
    for i, group1 in enumerate(groups):
        for group2 in groups[i+1:]:
            group1_data = df_clean[df_clean[demographic_col] == group1][target_col]
            group2_data = df_clean[df_clean[demographic_col] == group2][target_col]
            
            # Check minimum group size
            if len(group1_data) < min_group_size or len(group2_data) < min_group_size:
                continue
            
            # Perform Mann-Whitney U test
            mw_statistic, mw_p_value = perform_mannwhitney_test(
                group1_data,
                group2_data,
                alternative='two-sided'
            )
            
            # Perform Welch's t-test
            welch_statistic, welch_p_value = perform_welch_ttest(
                group1_data,
                group2_data,
                alternative='two-sided'
            )
            
            # Calculate descriptive statistics
            group1_mean = group1_data.mean()
            group2_mean = group2_data.mean()
            mean_diff = group1_mean - group2_mean
            
            group1_median = group1_data.median()
            group2_median = group2_data.median()
            median_diff = group1_median - group2_median
            
            # Check if tests agree
            mw_sig = mw_p_value < significance_level if not np.isnan(mw_p_value) else False
            welch_sig = welch_p_value < significance_level if not np.isnan(welch_p_value) else False
            tests_agree = mw_sig == welch_sig
            
            results.append({
                'demographic': demographic_col,
                'group1': group1,
                'group2': group2,
                'group1_mean': group1_mean,
                'group2_mean': group2_mean,
                'mean_difference': mean_diff,
                'group1_median': group1_median,
                'group2_median': group2_median,
                'median_difference': median_diff,
                'group1_count': len(group1_data),
                'group2_count': len(group2_data),
                'mw_statistic': mw_statistic,
                'mw_p_value': mw_p_value,
                'mw_significant': mw_sig,
                'welch_statistic': welch_statistic,
                'welch_p_value': welch_p_value,
                'welch_significant': welch_sig,
                'tests_agree': tests_agree
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Sort by Mann-Whitney p-value
        results_df = results_df.sort_values('mw_p_value')
    
    return results_df

