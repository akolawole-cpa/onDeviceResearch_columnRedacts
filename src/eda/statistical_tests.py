"""
Statistical Tests Module

Functions for performing statistical hypothesis tests and analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster

from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, norm
from typing import List, Dict, Optional, Tuple

try:
    from statsmodels.stats.proportion import proportions_ztest

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def aggregate_user_level_proportions(
    df: pd.DataFrame,
    feature_set: List[str],
    group_var: str = "wonky_study_count",
    user_id_var: str = "respondentPk",
) -> pd.DataFrame:
    """
    Aggregate user x task data to user-level proportions.

    For each user, calculates the proportion of tasks where each binary feature = 1.
    This restores statistical independence for hypothesis testing.

    Parameters:
    -----------
    df : pd.DataFrame
        User x task level data
    feature_set : List[str]
        Binary features to aggregate
    group_var : str
        Group indicator (wonky_study_count)
    user_id_var : str
        User identifier column

    Returns:
    --------
    pd.DataFrame
        One row per user with their proportion for each feature and group_binary
    """
    df_copy = df.copy()
    df_copy[group_var] = df_copy[group_var].fillna(0)
    df_copy["group_binary"] = (df_copy[group_var] > 0).astype(int)

    # Columns to aggregate: features + group_binary
    cols_to_agg = [f for f in feature_set if f in df_copy.columns]

    # For each user, calculate mean of each feature (proportion where feature=1)
    # and take max of group_binary (if user has any wonky tasks, they're in wonky group)
    agg_dict = {col: "mean" for col in cols_to_agg}
    agg_dict["group_binary"] = "max"

    user_df = df_copy.groupby(user_id_var).agg(agg_dict).reset_index()

    return user_df


def _interpret_effect_size(value: float, thresholds: Dict[str, float]) -> str:
    """
    Interpret effect size magnitude based on thresholds.

    Parameters:
    -----------
    value : float
        The effect size value (absolute value will be used)
    thresholds : Dict[str, float]
        Dictionary with keys 'small', 'medium', 'large' and their threshold values

    Returns:
    --------
    str
        Interpretation: 'negligible', 'small', 'medium', or 'large'
    """
    if pd.isna(value):
        return "undefined"

    abs_value = abs(value)

    if abs_value < thresholds["small"]:
        return "negligible"
    elif abs_value < thresholds["medium"]:
        return "small"
    elif abs_value < thresholds["large"]:
        return "medium"
    else:
        return "large"


def perform_chi_square_tests(
    df: pd.DataFrame,
    feature_set: List[str],
    group_var: str = "wonky_study_count",
    significance_level: float = 0.01,
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
        DataFrame with chi2, p_value, cramers_v, effect_size_interpretation,
        and significance for each feature
    """
    results = []

    # Cramér's V thresholds for effect size interpretation
    cramers_v_thresholds = {"small": 0.1, "medium": 0.3, "large": 0.5}

    # Prepare group variable (binary: >0 vs =0 or NaN)
    if group_var in df.columns:
        df_test = df.copy()
        df_test[group_var] = df_test[group_var].fillna(0)
        df_test["group_binary"] = (df_test[group_var] > 0).astype(int)
    else:
        return pd.DataFrame()

    for feature in feature_set:
        if feature not in df_test.columns:
            continue

        # Create contingency table
        contingency_table = pd.crosstab(df_test["group_binary"], df_test[feature])

        # Check if we have enough data
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            continue

        # Perform chi-squared test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            # Calculate Cramér's V effect size
            # V = sqrt(chi2 / (n * (min(r, c) - 1)))
            n = contingency_table.values.sum()
            min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else np.nan

            # Interpret effect size
            effect_interp = _interpret_effect_size(cramers_v, cramers_v_thresholds)

            results.append(
                {
                    "feature": feature,
                    "chi2": chi2,
                    "chi_p_value": p_value,
                    "dof": dof,
                    "cramers_v": cramers_v,
                    "effect_size_interpretation": effect_interp,
                    "significant": p_value < significance_level,
                }
            )
        except Exception as e:
            continue

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        results_df = results_df.set_index("feature")
        results_df = results_df.sort_values("chi2", ascending=False)

    return results_df


def perform_two_proportion_z_tests(
    df: pd.DataFrame,
    feature_set: List[str],
    group_var: str = "wonky_study_count",
    significance_level: float = 0.01,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """
    Perform two-proportion z-tests on a set of binary features comparing two groups.

    Tests whether binary features (proportions) differ significantly between groups (wonky vs non-wonky).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and group variable
    feature_set : List[str]
        List of binary feature column names to test (should be 0/1 or boolean)
    group_var : str
        Column name for grouping variable
    significance_level : float
        Significance level for determining significance
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', or 'greater'

    Returns:
    --------
    pd.DataFrame
        DataFrame with z_statistic, p_value, cohens_h, effect_size_interpretation,
        and significance for each feature.
        Indexed by feature name, sorted by absolute z_statistic (descending).
    """
    results = []

    # Cohen's h thresholds for effect size interpretation
    cohens_h_thresholds = {"small": 0.2, "medium": 0.5, "large": 0.8}

    # Prepare group variable (binary: >0 vs =0 or NaN)
    if group_var not in df.columns:
        return pd.DataFrame()

    df_test = df.copy()
    df_test[group_var] = df_test[group_var].fillna(0)
    df_test["group_binary"] = (df_test[group_var] > 0).astype(int)

    group1_mask = df_test["group_binary"] == 1
    group2_mask = df_test["group_binary"] == 0

    group1_total = group1_mask.sum()
    group2_total = group2_mask.sum()

    if group1_total == 0 or group2_total == 0:
        return pd.DataFrame()

    for feature in feature_set:
        if feature not in df_test.columns:
            continue

        # Get counts for each group
        group1_with_feature = (group1_mask & (df_test[feature] == 1)).sum()
        group2_with_feature = (group2_mask & (df_test[feature] == 1)).sum()

        # Calculate proportions
        group1_prop = group1_with_feature / group1_total if group1_total > 0 else 0.0
        group2_prop = group2_with_feature / group2_total if group2_total > 0 else 0.0

        # Perform two-proportion z-test
        try:
            if HAS_STATSMODELS:
                # Use statsmodels (more robust, handles edge cases better)
                try:
                    counts = np.array([group1_with_feature, group2_with_feature])
                    nobs = np.array([group1_total, group2_total])
                    z_stat, p_value = proportions_ztest(
                        counts, nobs, alternative=alternative
                    )
                except Exception as e:
                    # Fall back to manual calculation if statsmodels fails
                    z_stat, p_value = perform_two_proportion_ztest(
                        group1_with_feature,
                        group1_total,
                        group2_with_feature,
                        group2_total,
                        alternative=alternative,
                    )
            else:
                # Manual calculation using normal approximation
                z_stat, p_value = perform_two_proportion_ztest(
                    group1_with_feature,
                    group1_total,
                    group2_with_feature,
                    group2_total,
                    alternative=alternative,
                )

            # Calculate Cohen's h effect size
            # h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
            if 0 <= group1_prop <= 1 and 0 <= group2_prop <= 1:
                phi1 = 2 * np.arcsin(np.sqrt(group1_prop))
                phi2 = 2 * np.arcsin(np.sqrt(group2_prop))
                cohens_h = phi1 - phi2
            else:
                cohens_h = np.nan

            # Interpret effect size
            effect_interp = _interpret_effect_size(cohens_h, cohens_h_thresholds)

            results.append(
                {
                    "feature": feature,
                    "z_statistic": z_stat,
                    "p_value": p_value,
                    "significant": p_value < significance_level,
                    "group1_proportion": group1_prop,
                    "group2_proportion": group2_prop,
                    "proportion_diff": group1_prop - group2_prop,
                    "cohens_h": cohens_h,
                    "effect_size_interpretation": effect_interp,
                }
            )
        except Exception as e:
            continue

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        results_df = results_df.set_index("feature")
        results_df = results_df.sort_values("z_statistic", key=abs, ascending=False)

    return results_df


def perform_welch_ttests_on_proportions(
    df: pd.DataFrame,
    feature_set: List[str],
    group_var: str = "wonky_study_count",
    user_id_var: str = "respondentPk",
    significance_level: float = 0.01,
    alternative: str = "two-sided",
    min_users_per_group: int = 5,
) -> pd.DataFrame:
    """
    Perform Welch's t-tests on user-level proportions.
    
    Flow:
    1. Aggregate to user level (each user's proportion for each feature)
    2. Run Welch's t-test comparing user-level proportions between groups
    3. Return results in same format as your z-test function
    
    Parameters:
    -----------
    df : pd.DataFrame
        User x task level data
    feature_set : List[str]
        Binary features to test
    group_var : str
        Group variable (wonky_study_count)
    user_id_var : str
        User identifier
    significance_level : float
        Alpha level for significance
    alternative : str
        'two-sided', 'less', or 'greater'
    min_users_per_group : int
        Minimum users needed per group to run test
    
    Returns:
    --------
    pd.DataFrame
        Results with t_statistic, p_value, cohens_d, effect_size_interpretation,
        and other metrics for each feature
    """

    # Cohen's d thresholds for effect size interpretation
    cohens_d_thresholds = {"small": 0.2, "medium": 0.5, "large": 0.8}

    # Step 1: Aggregate to user level
    print(f"Aggregating {len(df)} user x task observations to user level...")
    user_df = aggregate_user_level_proportions(
        df, feature_set, group_var, user_id_var
    )
    print(f"Aggregated to {len(user_df)} unique users")

    # Separate groups
    group1_mask = user_df['group_binary'] == 1
    group2_mask = user_df['group_binary'] == 0

    n_group1 = group1_mask.sum()
    n_group2 = group2_mask.sum()

    print(f"Group 1 (wonky): {n_group1} users")
    print(f"Group 2 (non-wonky): {n_group2} users")

    # Check minimum sample size
    if n_group1 < min_users_per_group or n_group2 < min_users_per_group:
        print(f"Warning: Insufficient users per group (need {min_users_per_group}+)")
        return pd.DataFrame()

    results = []

    # Step 2: Run t-test for each feature
    for feature in feature_set:
        if feature not in user_df.columns:
            continue

        # Get user-level proportions for each group
        group1_props = user_df.loc[group1_mask, feature].dropna()
        group2_props = user_df.loc[group2_mask, feature].dropna()

        # Check we have data
        if len(group1_props) < 2 or len(group2_props) < 2:
            continue

        try:
            # Welch's t-test (unequal variances)
            if alternative == "two-sided":
                t_stat, p_value = stats.ttest_ind(
                    group1_props, group2_props, equal_var=False
                )
            elif alternative == "greater":
                # Group 1 > Group 2
                t_stat, p_value_two = stats.ttest_ind(
                    group1_props, group2_props, equal_var=False
                )
                p_value = p_value_two / 2 if t_stat > 0 else 1 - (p_value_two / 2)
            elif alternative == "less":
                # Group 1 < Group 2
                t_stat, p_value_two = stats.ttest_ind(
                    group1_props, group2_props, equal_var=False
                )
                p_value = p_value_two / 2 if t_stat < 0 else 1 - (p_value_two / 2)
            else:
                raise ValueError(f"Unknown alternative: {alternative}")

            # Calculate means
            mean_group1 = group1_props.mean()
            mean_group2 = group2_props.mean()
            mean_diff = mean_group1 - mean_group2

            # Effect size (Cohen's d) with proper pooled SD weighting
            n1, n2 = len(group1_props), len(group2_props)
            pooled_std = np.sqrt(
                ((n1 - 1) * group1_props.var() + (n2 - 1) * group2_props.var()) / (n1 + n2 - 2)
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

            # Interpret effect size
            effect_interp = _interpret_effect_size(cohens_d, cohens_d_thresholds)

            results.append({
                'feature': feature,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < significance_level,
                'group1_proportion': mean_group1,
                'group2_proportion': mean_group2,
                'proportion_diff': mean_diff,
                'cohens_d': cohens_d,
                'effect_size_interpretation': effect_interp,
                'n_users_group1': n1,
                'n_users_group2': n2,
            })

        except Exception as e:
            print(f"Error testing {feature}: {str(e)}")
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        results_df = results_df.set_index('feature')
        results_df = results_df.sort_values('t_statistic', key=abs, ascending=False)

    return results_df


def OLS_with_cluster_robust_test(
    df: pd.DataFrame,
    feature_set: List[str],
    outcome_var: str = "wonky_study_count",
    user_id_var: str = "respondentPk",
    significance_level: float = 0.01,
) -> pd.DataFrame:
    """
    OLS analysis with cluster-robust standard errors that account for within-user correlation.

    For binary features, also calculates Cohen's d effect size.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features, outcome, and user identifier
    feature_set : List[str]
        List of feature column names to test
    outcome_var : str
        Outcome variable name
    user_id_var : str
        User identifier for clustering
    significance_level : float
        Significance level for determining significance

    Returns:
    --------
    pd.DataFrame
        DataFrame with mean_difference, t_statistic, p_value, cohens_d,
        effect_size_interpretation, and significance for each feature
    """

    # Cohen's d thresholds for effect size interpretation
    cohens_d_thresholds = {"small": 0.2, "medium": 0.5, "large": 0.8}

    results = []

    for feature in feature_set:
        if feature not in df.columns:
            continue

        df_test = df[[user_id_var, outcome_var, feature]].copy()
        df_test = df_test.dropna()

        X = sm.add_constant(df_test[[feature]])
        y = df_test[outcome_var]

        model = sm.OLS(y, X)

        result_robust = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': df_test[user_id_var]}
        )

        coef = result_robust.params[feature]
        se = result_robust.bse[feature]
        t_stat = result_robust.tvalues[feature]
        p_value = result_robust.pvalues[feature]

        mean_with = df_test[df_test[feature] == 1][outcome_var].mean()
        mean_without = df_test[df_test[feature] == 0][outcome_var].mean()

        # Calculate Cohen's d effect size
        # d = coefficient / std(y) for binary predictors
        std_y = df_test[outcome_var].std()
        cohens_d = coef / std_y if std_y > 0 else np.nan

        # Interpret effect size
        effect_interp = _interpret_effect_size(cohens_d, cohens_d_thresholds)

        results.append({
            'feature': feature,
            'mean_with_feature': mean_with,
            'mean_without_feature': mean_without,
            'mean_difference': coef,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < significance_level,
            'se_cluster_robust': se,
            'cohens_d': cohens_d,
            'effect_size_interpretation': effect_interp,
        })

    return pd.DataFrame(results).set_index('feature').sort_values('t_statistic', key=abs, ascending=False)