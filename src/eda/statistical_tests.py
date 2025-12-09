"""
Statistical Tests Module

Functions for performing statistical hypothesis tests and analysis.
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, norm
from typing import List, Dict, Optional, Tuple

# Try to import statsmodels for two-proportion z-test (more robust)
try:
    from statsmodels.stats.proportion import proportions_ztest
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def perform_mannwhitney_test(
    group1: pd.Series,
    group2: pd.Series,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test (non-parametric test for two independent samples).
    
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
    Tuple[float, float]
        (statistic, p_value)
    """
    statistic, p_value = mannwhitneyu(
        group1.dropna(),
        group2.dropna(),
        alternative=alternative
    )
    
    return statistic, p_value


def perform_welch_ttest(
    group1: pd.Series,
    group2: pd.Series,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform Welch's t-test (parametric test that doesn't assume equal variances).
    
    Parameters:
    -----------
    group1 : pd.Series
        First group data
    group2 : pd.Series
        Second group data
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', or 'greater'
        Note: Older scipy versions may not support 'alternative' parameter
        
    Returns:
    --------
    Tuple[float, float]
        (statistic, p_value)
    """
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()
    
    if len(group1_clean) == 0 or len(group2_clean) == 0:
        return np.nan, np.nan
    
    # Try with alternative parameter (newer scipy)
    try:
        statistic, p_value = ttest_ind(
            group1_clean,
            group2_clean,
            equal_var=False,  # Welch's t-test
            alternative=alternative
        )
    except TypeError:
        # Fallback for older scipy versions
        statistic, p_value = ttest_ind(
            group1_clean,
            group2_clean,
            equal_var=False
        )
        # Adjust p-value for one-sided tests if needed
        if alternative == 'less':
            p_value = p_value / 2 if statistic < 0 else 1 - p_value / 2
        elif alternative == 'greater':
            p_value = p_value / 2 if statistic > 0 else 1 - p_value / 2
    
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
    Compare two groups statistically using Mann-Whitney U test and optionally Welch's t-test.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the groups and metrics
    group_col : str
        Column name for group labels
    metrics : List[str]
        List of metric column names to compare
    group1_value : int
        Value for first group (default: 1)
    group2_value : int
        Value for second group (default: 0)
    significance_level : float
        Significance level for determining significance
    include_welch : bool
        Whether to include Welch's t-test results
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with test results for each metric
    """
    results = []
    
    group1 = df[df[group_col] == group1_value]
    group2 = df[df[group_col] == group2_value]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        group1_data = group1[metric].dropna()
        group2_data = group2[metric].dropna()
        
        if len(group1_data) == 0 or len(group2_data) == 0:
            continue
        
        # Mann-Whitney U test
        mw_statistic, mw_p_value = perform_mannwhitney_test(group1_data, group2_data)
        
        # Calculate descriptive statistics
        result = {
            'metric': metric,
            'group1_mean': group1_data.mean(),
            'group2_mean': group2_data.mean(),
            'mean_difference': group1_data.mean() - group2_data.mean(),
            'group1_median': group1_data.median(),
            'group2_median': group2_data.median(),
            'group1_count': len(group1_data),
            'group2_count': len(group2_data),
            'mw_statistic': mw_statistic,
            'mw_p_value': mw_p_value,
            'mw_significant': mw_p_value < significance_level
        }
        
        # Add Welch's t-test if requested
        if include_welch:
            welch_statistic, welch_p_value = perform_welch_ttest(group1_data, group2_data)
            result['welch_statistic'] = welch_statistic
            result['welch_p_value'] = welch_p_value
            result['welch_significant'] = welch_p_value < significance_level if not np.isnan(welch_p_value) else False
            result['tests_agree'] = result['mw_significant'] == result['welch_significant']
        
        results.append(result)
    
    return pd.DataFrame(results)


def compare_groups_with_both_tests(
    df: pd.DataFrame,
    group_col: str,
    metrics: List[str],
    group1_value: int = 1,
    group2_value: int = 0,
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    Convenience function to run both Mann-Whitney U and Welch's t-test.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the groups and metrics
    group_col : str
        Column name for group labels
    metrics : List[str]
        List of metric column names to compare
    group1_value : int
        Value for first group (default: 1)
    group2_value : int
        Value for second group (default: 0)
    significance_level : float
        Significance level for determining significance
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with test results for each metric (includes both tests)
    """
    return compare_groups_statistically(
        df, group_col, metrics, group1_value, group2_value, significance_level, include_welch=True
    )


def analyze_thresholds(
    df: pd.DataFrame,
    feature: str,
    bins: List[float],
    target_col: str = "has_wonky_tasks"
) -> pd.DataFrame:
    """
    Analyze how wonky task rates vary across feature thresholds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the feature and target
    feature : str
        Feature column name to analyze
    bins : List[float]
        List of bin edges for threshold analysis
    target_col : str
        Target column name (binary: 1 = wonky, 0 = non-wonky)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with threshold analysis results
    """
    df_clean = df[[feature, target_col]].dropna()
    
    # Create bins
    df_clean['bin'] = pd.cut(df_clean[feature], bins=bins, include_lowest=True)
    
    # Calculate wonky rates by bin
    results = []
    for bin_val in df_clean['bin'].cat.categories:
        bin_data = df_clean[df_clean['bin'] == bin_val]
        if len(bin_data) == 0:
            continue
        
        wonky_rate = bin_data[target_col].mean()
        total_count = len(bin_data)
        wonky_count = bin_data[target_col].sum()
        
        results.append({
            'bin': str(bin_val),
            'total_count': total_count,
            'wonky_count': wonky_count,
            'wonky_rate': wonky_rate
        })
    
    return pd.DataFrame(results)


def format_hypothesis_results(
    results_df: pd.DataFrame,
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    Format hypothesis test results for display.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with test results
    significance_level : float
        Significance level used
        
    Returns:
    --------
    pd.DataFrame
        Formatted results DataFrame
    """
    results = results_df.copy()
    
    # Round numeric columns
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    results[numeric_cols] = results[numeric_cols].round(4)
    
    # Format p-values
    if 'mw_p_value' in results.columns:
        results['mw_p_value'] = results['mw_p_value'].apply(
            lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"
        )
    
    if 'welch_p_value' in results.columns:
        results['welch_p_value'] = results['welch_p_value'].apply(
            lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"
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


def compare_task_category_wonky_rates(
    df: pd.DataFrame,
    category_col: str = "defined_task_category",
    respondent_id_col: str = "respondentPk",
    wonky_col: str = "wonky_task_instances"
) -> pd.DataFrame:
    """
    Create summary table showing wonky rates by task category.
    
    For each task category, calculates:
    - Count of unique respondents with wonky_task_instances = 0 or NA (non-wonky)
    - Count of unique respondents with wonky_task_instances > 0 (wonky)
    - Percentage of wonky respondents
    - Delta (difference) between wonky and non-wonky proportions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Task-level DataFrame with category column, respondent ID, and wonky_task_instances column
    category_col : str
        Column name for task category (default: "defined_task_category")
    respondent_id_col : str
        Column name for respondent ID (default: "respondentPk")
    wonky_col : str
        Column name for wonky instances (default: "wonky_task_instances")
        
    Returns:
    --------
    pd.DataFrame
        Summary table with columns:
        - defined_task_category: task category
        - non_wonky_count: count of unique respondents with wonky_task_instances = 0 or NA
        - wonky_count: count of unique respondents with wonky_task_instances > 0
        - total_respondents: total unique respondents for this category
        - wonky_pct: percentage of respondents with wonky instances
        - non_wonky_pct: percentage of respondents without wonky instances
        - proportion_delta: difference between wonky_pct and non_wonky_pct
    """
    # Check required columns exist
    required_cols = [category_col, respondent_id_col, wonky_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get unique respondents per category with their wonky status
    # For each respondent-category combination, get the wonky_task_instances value
    # We'll use the first non-null value per respondent-category, or 0 if all are null
    category_respondent_wonky = (
        df[[category_col, respondent_id_col, wonky_col]]
        .drop_duplicates(subset=[category_col, respondent_id_col])
        .copy()
    )
    
    # Fill NaN with 0 for counting purposes
    category_respondent_wonky[wonky_col] = category_respondent_wonky[wonky_col].fillna(0)
    
    # First, get unique respondents per category with their wonky status
    # For each respondent-category combination, determine if they have any wonky instances > 0
    respondent_category_wonky = (
        category_respondent_wonky.groupby([category_col, respondent_id_col])[wonky_col]
        .max()  # If any task has wonky > 0, max will be > 0
        .reset_index()
    )
    
    # Create binary wonky flag
    respondent_category_wonky['is_wonky'] = (respondent_category_wonky[wonky_col] > 0).astype(int)
    
    # Calculate total rows (respondent-category pairs) for wonky and non-wonky across ALL categories
    # Count rows, not unique respondents (a respondent can appear in multiple categories)
    total_wonky_rows = (respondent_category_wonky['is_wonky'] == 1).sum()
    total_non_wonky_rows = (respondent_category_wonky['is_wonky'] == 0).sum()
    
    # Group by category and calculate counts
    results = []
    
    for category in category_respondent_wonky[category_col].unique():
        category_data = respondent_category_wonky[
            respondent_category_wonky[category_col] == category
        ]
        
        # Count rows (respondent-category pairs) in this category
        non_wonky_count = (category_data['is_wonky'] == 0).sum()
        wonky_count = (category_data['is_wonky'] == 1).sum()
        total_respondents = len(category_data)
        
        # Calculate proportions ACROSS groups (not within category)
        # wonky_pct = proportion of all wonky rows that are in this category
        # non_wonky_pct = proportion of all non-wonky rows that are in this category
        if total_wonky_rows > 0:
            wonky_pct = (wonky_count / total_wonky_rows) * 100
        else:
            wonky_pct = 0.0
        
        if total_non_wonky_rows > 0:
            non_wonky_pct = (non_wonky_count / total_non_wonky_rows) * 100
        else:
            non_wonky_pct = 0.0
        
        # Delta is the difference between proportions
        proportion_delta = wonky_pct - non_wonky_pct
        
        results.append({
            'defined_task_category': category,
            'non_wonky_count': non_wonky_count,
            'wonky_count': wonky_count,
            'total_respondents': total_respondents,
            'wonky_pct': wonky_pct,
            'non_wonky_pct': non_wonky_pct,
            'proportion_delta': proportion_delta
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by total_respondents descending
    results_df = results_df.sort_values('total_respondents', ascending=False).reset_index(drop=True)
    
    return results_df


def compare_speed_categories_proportions(
    df: pd.DataFrame,
    speed_features: List[str],
    group_col: str = "wonky_task_instances",
    group_threshold: float = 0,
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    Compare proportions of speed categories between wonky and non-wonky groups using two-proportion z-test.
    
    For each speed category (fast, normal, slow, etc.), tests whether the proportion differs
    significantly between wonky and non-wonky task instances.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with speed features and group column
    speed_features : List[str]
        List of speed feature column names (e.g., ['is_fast', 'is_normal_speed', 'is_slow'])
    group_col : str
        Column name for grouping variable (default: "wonky_task_instances")
    group_threshold : float
        Threshold for determining wonky vs non-wonky groups
    significance_level : float
        Significance level for determining significance
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with comparison results for each speed category:
        - feature: speed category name
        - wonky_proportion: proportion in wonky group
        - non_wonky_proportion: proportion in non-wonky group
        - proportion_diff: difference (wonky - non_wonky)
        - z_statistic: z-statistic from two-proportion z-test
        - p_value: p-value from test
        - significant: whether p-value < significance_level
        - wonky_count: number of wonky tasks with this speed category
        - wonky_total: total number of wonky tasks
        - non_wonky_count: number of non-wonky tasks with this speed category
        - non_wonky_total: total number of non-wonky tasks
    """
    if group_col not in df.columns:
        return pd.DataFrame()
    
    # Prepare group variable
    df_test = df.copy()
    
    # Create binary group variable
    wonky_mask = df_test[group_col] > group_threshold
    
    # Handle non-wonky: check if column has NaN values
    if df_test[group_col].isna().sum() > 0:
        non_wonky_mask = df_test[group_col].isna()
    else:
        non_wonky_mask = df_test[group_col] == 0
    
    results = []
    
    for feature in speed_features:
        if feature not in df_test.columns:
            continue
        
        # Get counts for each group
        wonky_total = wonky_mask.sum()
        non_wonky_total = non_wonky_mask.sum()
        
        if wonky_total == 0 or non_wonky_total == 0:
            continue
        
        # Count how many in each group have this speed category
        wonky_with_feature = (wonky_mask & (df_test[feature] == 1)).sum()
        non_wonky_with_feature = (non_wonky_mask & (df_test[feature] == 1)).sum()
        
        # Calculate proportions
        wonky_prop = wonky_with_feature / wonky_total if wonky_total > 0 else 0.0
        non_wonky_prop = non_wonky_with_feature / non_wonky_total if non_wonky_total > 0 else 0.0
        prop_diff = wonky_prop - non_wonky_prop
        
        # Perform two-proportion z-test
        # H0: p1 = p2 (proportions are equal)
        # H1: p1 != p2 (proportions differ)
        
        if HAS_STATSMODELS:
            # Use statsmodels (more robust, handles edge cases better)
            try:
                counts = np.array([wonky_with_feature, non_wonky_with_feature])
                nobs = np.array([wonky_total, non_wonky_total])
                z_stat, p_value = proportions_ztest(counts, nobs, alternative='two-sided')
            except Exception as e:
                # Fall back to manual calculation if statsmodels fails
                z_stat, p_value = _manual_two_proportion_ztest(
                    wonky_with_feature, wonky_total,
                    non_wonky_with_feature, non_wonky_total
                )
        else:
            # Manual calculation using normal approximation
            z_stat, p_value = _manual_two_proportion_ztest(
                wonky_with_feature, wonky_total,
                non_wonky_with_feature, non_wonky_total
            )
        
        results.append({
            'feature': feature,
            'wonky_proportion': wonky_prop,
            'non_wonky_proportion': non_wonky_prop,
            'proportion_diff': prop_diff,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < significance_level,
            'wonky_count': wonky_with_feature,
            'wonky_total': wonky_total,
            'non_wonky_count': non_wonky_with_feature,
            'non_wonky_total': non_wonky_total
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Sort by absolute z-statistic (most significant first)
        results_df = results_df.sort_values('z_statistic', key=abs, ascending=False)
    
    return results_df


def _manual_two_proportion_ztest(
    count1: int, nobs1: int,
    count2: int, nobs2: int
) -> Tuple[float, float]:
    """
    Manual two-proportion z-test calculation.
    
    Parameters:
    -----------
    count1 : int
        Number of successes in group 1
    nobs1 : int
        Total number of observations in group 1
    count2 : int
        Number of successes in group 2
    nobs2 : int
        Total number of observations in group 2
        
    Returns:
    --------
    Tuple[float, float]
        (z-statistic, p-value)
    """
    # Calculate proportions
    p1 = count1 / nobs1 if nobs1 > 0 else 0.0
    p2 = count2 / nobs2 if nobs2 > 0 else 0.0
    
    # Pooled proportion (under null hypothesis that p1 = p2)
    p_pooled = (count1 + count2) / (nobs1 + nobs2) if (nobs1 + nobs2) > 0 else 0.0
    
    # Standard error of the difference
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/nobs1 + 1/nobs2))
    
    # Avoid division by zero
    if se == 0:
        return (0.0, 1.0)
    
    # Z-statistic
    z_stat = (p1 - p2) / se
    
    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    return (z_stat, p_value)
