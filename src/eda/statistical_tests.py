"""
Statistical Tests Module - Simplified & Optimized

Contains only the regression functions actually used in the analysis pipeline:
- OLS_with_cluster_robust_test (parallelized)
- logistic_regression_with_cluster_robust_test (parallelized)
- run_combined_regression_tests (parallelized)

All functions support parallel execution via joblib for significant speedups
when testing many features (e.g., 281 features -> ~4-6x faster with n_jobs=-1).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Logit

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("joblib not installed. Install with: pip install joblib")


# =============================================================================
# Effect Size Interpretation (Internal)
# =============================================================================

def _interpret_effect_size(value: float, thresholds: Dict[str, float]) -> str:
    """Interpret effect size magnitude based on thresholds."""
    if pd.isna(value):
        return "undefined"

    abs_value = abs(value)

    if abs_value < thresholds["small"]:
        return "negligible"
    elif abs_value < thresholds["medium"]:
        return "small"
    elif abs_value < thresholds["large"]:
        return "medium"
    return "large"


def _interpret_effect_size_or(value: float, thresholds: Dict[str, float]) -> str:
    """Interpret effect size for odds ratios (distance from 1)."""
    if pd.isna(value) or value <= 0:
        return "undefined"
    
    if value < thresholds["small"]:
        return "negligible"
    elif value < thresholds["medium"]:
        return "small"
    elif value < thresholds["large"]:
        return "medium"
    return "large"


# =============================================================================
# Single Feature Fitting (Internal - for parallel execution)
# =============================================================================

def _fit_ols_single_feature(
    feature: str,
    df: pd.DataFrame,
    outcome_var: str,
    user_id_var: str,
    significance_level: float,
    cohens_d_thresholds: Dict[str, float],
) -> Optional[Dict]:
    """Fit OLS with cluster-robust SE for a single feature."""
    if feature not in df.columns:
        return None
    
    cols_needed = [user_id_var, outcome_var, feature]
    df_test = df[cols_needed].dropna()
    
    if len(df_test) < 30 or df_test[feature].nunique() < 2:
        return None
    
    try:
        X = sm.add_constant(df_test[[feature]])
        y = df_test[outcome_var]
        
        model = OLS(y, X)
        result = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': df_test[user_id_var]}
        )
        
        coef = result.params[feature]
        mask_with = df_test[feature] == 1
        std_y = df_test[outcome_var].std()
        cohens_d = coef / std_y if std_y > 0 else np.nan
        
        return {
            'feature': feature,
            'mean_with_feature': df_test.loc[mask_with, outcome_var].mean(),
            'mean_without_feature': df_test.loc[~mask_with, outcome_var].mean(),
            'mean_difference': coef,
            't_statistic': result.tvalues[feature],
            'p_value': result.pvalues[feature],
            'significant': result.pvalues[feature] < significance_level,
            'se_cluster_robust': result.bse[feature],
            'cohens_d': cohens_d,
            'effect_size_interpretation': _interpret_effect_size(cohens_d, cohens_d_thresholds),
        }
    except Exception:
        return None


def _fit_logit_single_feature(
    feature: str,
    df: pd.DataFrame,
    outcome_col: str,
    user_id_var: str,
    significance_level: float,
    or_thresholds: Dict[str, float],
) -> Optional[Dict]:
    """Fit Logistic regression with cluster-robust SE for a single feature."""
    if feature not in df.columns:
        return None
    
    cols_needed = [user_id_var, outcome_col, feature]
    df_test = df[cols_needed].dropna()
    
    if len(df_test) < 30:
        return None
    
    y = df_test[outcome_col]
    if y.nunique() < 2 or df_test[feature].nunique() < 2:
        return None
    
    try:
        X = sm.add_constant(df_test[[feature]])
        model = Logit(y, X)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(
                cov_type='cluster',
                cov_kwds={'groups': df_test[user_id_var]},
                disp=False,
                maxiter=100,
            )
        
        coef = result.params[feature]
        conf_int = result.conf_int().loc[feature]
        odds_ratio = np.exp(coef)
        
        mask_with = df_test[feature] == 1
        or_effect = max(odds_ratio, 1/odds_ratio) if odds_ratio > 0 else np.nan
        
        return {
            'feature': feature,
            'log_odds_coef': coef,
            'odds_ratio': odds_ratio,
            'or_ci_lower': np.exp(conf_int[0]),
            'or_ci_upper': np.exp(conf_int[1]),
            'z_statistic': result.tvalues[feature],
            'p_value': result.pvalues[feature],
            'se_cluster_robust': result.bse[feature],
            'significant': result.pvalues[feature] < significance_level,
            'prop_outcome_with_feature': df_test.loc[mask_with, outcome_col].mean(),
            'prop_outcome_without_feature': df_test.loc[~mask_with, outcome_col].mean(),
            'prop_difference': df_test.loc[mask_with, outcome_col].mean() - df_test.loc[~mask_with, outcome_col].mean(),
            'pseudo_r2': result.prsquared,
            'effect_size_interpretation': _interpret_effect_size_or(or_effect, or_thresholds),
            'n_obs': len(df_test),
        }
    except Exception:
        return None


# =============================================================================
# Public API
# =============================================================================

def OLS_with_cluster_robust_test(
    df: pd.DataFrame,
    feature_set: List[str],
    outcome_var: str = "wonky_study_count",
    user_id_var: str = "respondentPk",
    significance_level: float = 0.05,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    OLS analysis with cluster-robust standard errors (parallelized).

    Parameters
    ----------
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
    n_jobs : int
        Number of parallel jobs. -1 = all cores, 1 = sequential

    Returns
    -------
    pd.DataFrame
        Results with mean_difference, t_statistic, p_value, cohens_d,
        effect_size_interpretation, and significance for each feature
    """
    cohens_d_thresholds = {"small": 0.2, "medium": 0.5, "large": 0.8}
    
    valid_features = [f for f in feature_set if f in df.columns]
    if not valid_features:
        return pd.DataFrame()
    
    # Prepare data once
    cols = [user_id_var, outcome_var] + valid_features
    df_prep = df[cols].copy()
    df_prep[outcome_var] = df_prep[outcome_var].fillna(0)
    
    # Parallel or sequential execution
    if HAS_JOBLIB and n_jobs != 1 and len(valid_features) > 3:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_ols_single_feature)(
                f, df_prep, outcome_var, user_id_var, significance_level, cohens_d_thresholds
            )
            for f in valid_features
        )
    else:
        results = [
            _fit_ols_single_feature(
                f, df_prep, outcome_var, user_id_var, significance_level, cohens_d_thresholds
            )
            for f in valid_features
        ]
    
    results = [r for r in results if r is not None]
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).set_index('feature').sort_values('t_statistic', key=abs, ascending=False)


def logistic_regression_with_cluster_robust_test(
    df: pd.DataFrame,
    feature_set: List[str],
    outcome_var: str = "wonky_study_count",
    user_id_var: str = "respondentPk",
    significance_level: float = 0.05,
    binarize_outcome: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Logistic regression with cluster-robust standard errors (parallelized).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features, outcome, and user identifier
    feature_set : List[str]
        List of feature column names to test
    outcome_var : str
        Outcome variable name (will be binarized if binarize_outcome=True)
    user_id_var : str
        User identifier for clustering
    significance_level : float
        Significance level for determining significance
    binarize_outcome : bool
        If True, convert outcome to binary (>0 = 1, else 0)
    n_jobs : int
        Number of parallel jobs. -1 = all cores, 1 = sequential

    Returns
    -------
    pd.DataFrame
        Results with odds_ratio, z_statistic, p_value, effect_size for each feature
        
    Notes
    -----
    - Odds Ratio > 1: feature increases probability of outcome
    - Odds Ratio < 1: feature decreases probability of outcome
    """
    or_thresholds = {"small": 1.5, "medium": 2.5, "large": 4.0}
    
    valid_features = [f for f in feature_set if f in df.columns]
    if not valid_features:
        return pd.DataFrame()
    
    # Prepare data once
    cols = [user_id_var, outcome_var] + valid_features
    df_prep = df[cols].copy()
    df_prep[outcome_var] = df_prep[outcome_var].fillna(0)
    
    outcome_col = outcome_var
    if binarize_outcome:
        outcome_col = f"{outcome_var}_binary"
        df_prep[outcome_col] = (df_prep[outcome_var] > 0).astype(int)
    
    if df_prep[outcome_col].nunique() < 2:
        warnings.warn("Outcome variable has no variation")
        return pd.DataFrame()
    
    # Parallel or sequential execution
    if HAS_JOBLIB and n_jobs != 1 and len(valid_features) > 3:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_logit_single_feature)(
                f, df_prep, outcome_col, user_id_var, significance_level, or_thresholds
            )
            for f in valid_features
        )
    else:
        results = [
            _fit_logit_single_feature(
                f, df_prep, outcome_col, user_id_var, significance_level, or_thresholds
            )
            for f in valid_features
        ]
    
    results = [r for r in results if r is not None]
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).set_index('feature').sort_values('z_statistic', key=abs, ascending=False)


def run_combined_regression_tests(
    df: pd.DataFrame,
    feature_set: List[str],
    outcome_var: str = "wonky_study_count",
    user_id_var: str = "respondentPk",
    significance_level: float = 0.05,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Run both OLS and Logistic regression and combine results (parallelized).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features, outcome, and user identifier
    feature_set : List[str]
        List of feature column names to test
    outcome_var : str
        Outcome variable name
    user_id_var : str
        User identifier for clustering
    significance_level : float
        Significance level
    n_jobs : int
        Number of parallel jobs. -1 = all cores, 1 = sequential

    Returns
    -------
    pd.DataFrame
        Combined results with OLS and Logistic metrics side by side,
        plus 'significant_both' and 'significant_either' columns
    """
    ols_results = OLS_with_cluster_robust_test(
        df, feature_set, outcome_var, user_id_var, significance_level, n_jobs=n_jobs
    )
    
    logit_results = logistic_regression_with_cluster_robust_test(
        df, feature_set, outcome_var, user_id_var, significance_level, n_jobs=n_jobs
    )
    
    if len(ols_results) == 0 and len(logit_results) == 0:
        return pd.DataFrame()
    
    # Prefix columns
    if len(ols_results) > 0:
        ols_results = ols_results.add_prefix('ols_')
    if len(logit_results) > 0:
        logit_results = logit_results.add_prefix('logit_')
    
    # Merge
    if len(ols_results) > 0 and len(logit_results) > 0:
        combined = ols_results.join(logit_results, how='outer')
    elif len(ols_results) > 0:
        combined = ols_results
    else:
        combined = logit_results
    
    # Add summary columns
    if 'ols_significant' in combined.columns and 'logit_significant' in combined.columns:
        combined['significant_both'] = combined['ols_significant'] & combined['logit_significant']
        combined['significant_either'] = combined['ols_significant'] | combined['logit_significant']
    
    if 'ols_t_statistic' in combined.columns:
        combined = combined.sort_values('ols_t_statistic', key=abs, ascending=False)

    return combined


# =============================================================================
# STAKEHOLDER-FRIENDLY FORMATTING FUNCTIONS
# =============================================================================

def format_ols_for_stakeholders(
    ols_results: pd.DataFrame,
    baseline_mean: float,
    confidence_level: float = 0.95,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """
    Format OLS results with stakeholder-friendly interpretations.

    Parameters
    ----------
    ols_results : pd.DataFrame
        Output from OLS_with_cluster_robust_test()
    baseline_mean : float
        Mean of outcome variable (for percentage calculations)
    confidence_level : float
        Confidence level for intervals (default 0.95)
    significance_level : float
        Threshold for significance stars

    Returns
    -------
    pd.DataFrame with columns:
        - feature: Feature name
        - coefficient: Raw coefficient value
        - ci_lower, ci_upper: Confidence interval bounds
        - ci_formatted: "coef [lower, upper]" string
        - pct_change: Coefficient as % of baseline
        - p_value: Raw p-value
        - significance: Stars (*** / ** / *)
        - interpretation: Plain English description
        - impact_category: HIGH/MEDIUM/LOW/NOT SIGNIFICANT
    """
    from scipy import stats

    results = ols_results.reset_index() if 'feature' not in ols_results.columns else ols_results.copy()

    # Calculate confidence intervals
    z_crit = stats.norm.ppf((1 + confidence_level) / 2)
    results['ci_lower'] = results['mean_difference'] - z_crit * results['se_cluster_robust']
    results['ci_upper'] = results['mean_difference'] + z_crit * results['se_cluster_robust']

    # Formatted CI string
    results['ci_formatted'] = results.apply(
        lambda r: f"{r['mean_difference']:+.3f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]",
        axis=1
    )

    # Percentage change from baseline
    if baseline_mean != 0:
        results['pct_change'] = (results['mean_difference'] / baseline_mean * 100).round(1)
    else:
        results['pct_change'] = np.nan

    # Significance stars
    def get_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return ''

    results['significance'] = results['p_value'].apply(get_stars)

    # Plain English interpretation
    def interpret_coefficient(row):
        if row['p_value'] >= significance_level:
            return "No significant effect"

        coef = row['mean_difference']
        pct = row['pct_change'] if pd.notna(row['pct_change']) else 0

        if coef > 0:
            return f"Increases wonkiness by {abs(pct):.0f}%"
        else:
            return f"Decreases wonkiness by {abs(pct):.0f}%"

    results['interpretation'] = results.apply(interpret_coefficient, axis=1)

    # Impact category based on Cohen's d
    def categorize_impact(row):
        if row['p_value'] >= significance_level:
            return "NOT SIGNIFICANT"
        d = abs(row.get('cohens_d', 0))
        if d >= 0.8:
            return "HIGH"
        elif d >= 0.5:
            return "MEDIUM"
        elif d >= 0.2:
            return "LOW"
        return "NEGLIGIBLE"

    results['impact_category'] = results.apply(categorize_impact, axis=1)

    # Select and reorder columns
    output_cols = [
        'feature', 'mean_difference', 'ci_lower', 'ci_upper', 'ci_formatted',
        'pct_change', 'p_value', 'significance', 'interpretation', 'impact_category'
    ]
    available_cols = [c for c in output_cols if c in results.columns]

    return results[available_cols].sort_values('mean_difference', key=abs, ascending=False)


def format_odds_ratios_for_stakeholders(
    logit_results: pd.DataFrame,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """
    Format odds ratios with stakeholder-friendly interpretations.

    Parameters
    ----------
    logit_results : pd.DataFrame
        Output from logistic_regression_with_cluster_robust_test()
    significance_level : float
        Threshold for significance

    Returns
    -------
    pd.DataFrame with columns:
        - feature: Feature name
        - odds_ratio: Raw OR value
        - or_formatted: "OR [CI lower, CI upper]" string
        - interpretation: Plain English (e.g., "50% more likely")
        - p_value: Raw p-value
        - significance: Stars
        - impact_category: HIGH/MEDIUM/LOW/NOT SIGNIFICANT
        - direction: "Risk Factor" / "Protective Factor" / "Neutral"
    """
    results = logit_results.reset_index() if 'feature' not in logit_results.columns else logit_results.copy()

    # Formatted OR with CI
    if 'or_ci_lower' in results.columns and 'or_ci_upper' in results.columns:
        results['or_formatted'] = results.apply(
            lambda r: f"{r['odds_ratio']:.2f} [{r['or_ci_lower']:.2f}, {r['or_ci_upper']:.2f}]",
            axis=1
        )
    else:
        results['or_formatted'] = results['odds_ratio'].apply(lambda x: f"{x:.2f}")

    # Significance stars
    def get_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return ''

    results['significance'] = results['p_value'].apply(get_stars)

    # Plain English interpretation
    def interpret_or(row):
        or_val = row['odds_ratio']
        p_val = row['p_value']

        if p_val >= significance_level:
            return "No significant effect"

        if pd.isna(or_val) or or_val <= 0:
            return "Invalid"

        if or_val > 100 or or_val < 0.01:
            return "Extreme value - interpret with caution"

        if abs(or_val - 1) < 0.05:
            return "No practical effect"

        if or_val > 1:
            pct = (or_val - 1) * 100
            if pct > 100:
                return f"{or_val:.1f}x more likely to be flagged"
            return f"{pct:.0f}% more likely to be flagged"
        else:
            pct = (1 - or_val) * 100
            return f"{pct:.0f}% less likely to be flagged"

    results['interpretation'] = results.apply(interpret_or, axis=1)

    # Impact category based on OR magnitude
    def categorize_impact(row):
        if row['p_value'] >= significance_level:
            return "NOT SIGNIFICANT"

        or_val = row['odds_ratio']
        if pd.isna(or_val) or or_val <= 0:
            return "INVALID"

        # Use max(OR, 1/OR) for symmetric interpretation
        or_effect = max(or_val, 1/or_val) if or_val > 0 else 1

        if or_effect >= 4.0:
            return "HIGH"
        elif or_effect >= 2.5:
            return "MEDIUM"
        elif or_effect >= 1.5:
            return "LOW"
        return "NEGLIGIBLE"

    results['impact_category'] = results.apply(categorize_impact, axis=1)

    # Direction classification
    def classify_direction(row):
        if row['p_value'] >= significance_level:
            return "Neutral"
        or_val = row['odds_ratio']
        if pd.isna(or_val):
            return "Unknown"
        if or_val > 1.05:
            return "Risk Factor"
        elif or_val < 0.95:
            return "Protective Factor"
        return "Neutral"

    results['direction'] = results.apply(classify_direction, axis=1)

    # Select and reorder columns
    output_cols = [
        'feature', 'odds_ratio', 'or_formatted', 'interpretation',
        'p_value', 'significance', 'impact_category', 'direction'
    ]
    available_cols = [c for c in output_cols if c in results.columns]

    return results[available_cols].sort_values('odds_ratio', key=lambda x: abs(np.log(x)), ascending=False)


def generate_testing_executive_summary(
    combined_results: pd.DataFrame,
    significance_level: float = 0.05,
) -> str:
    """
    Generate executive summary of statistical testing results.

    Parameters
    ----------
    combined_results : pd.DataFrame
        Output from run_combined_regression_tests()
    significance_level : float
        Threshold for significance

    Returns
    -------
    str
        Formatted executive summary
    """
    results = combined_results.reset_index() if combined_results.index.name else combined_results.copy()

    summary_parts = []

    # Header
    summary_parts.append("=" * 70)
    summary_parts.append("EXECUTIVE SUMMARY: Statistical Testing Results")
    summary_parts.append("=" * 70)

    # Overview
    n_features = len(results)
    n_sig_ols = (results['ols_significant'] == True).sum() if 'ols_significant' in results.columns else 0
    n_sig_logit = (results['logit_significant'] == True).sum() if 'logit_significant' in results.columns else 0
    n_sig_both = (results['significant_both'] == True).sum() if 'significant_both' in results.columns else 0

    summary_parts.append(f"\nOVERVIEW:")
    summary_parts.append(f"  Total features tested: {n_features}")
    summary_parts.append(f"  Significant in OLS: {n_sig_ols} ({n_sig_ols/n_features*100:.1f}%)")
    summary_parts.append(f"  Significant in Logistic: {n_sig_logit} ({n_sig_logit/n_features*100:.1f}%)")
    summary_parts.append(f"  Significant in BOTH (high confidence): {n_sig_both}")

    # Top Risk Factors
    summary_parts.append("\n" + "-" * 50)
    summary_parts.append("TOP 5 RISK FACTORS (Increase Likelihood of Flagging)")
    summary_parts.append("-" * 50)

    if 'logit_odds_ratio' in results.columns and 'logit_significant' in results.columns:
        sig_results = results[results['logit_significant'] == True]
        top_risk = sig_results[sig_results['logit_odds_ratio'] > 1].nlargest(5, 'logit_odds_ratio')

        for i, (_, row) in enumerate(top_risk.iterrows(), 1):
            or_val = row['logit_odds_ratio']
            pct = (or_val - 1) * 100
            feature = row['feature'] if 'feature' in row else row.name
            summary_parts.append(
                f"  {i}. {feature}: {pct:.0f}% more likely (OR={or_val:.2f}, p={row['logit_p_value']:.4f})"
            )

    # Top Protective Factors
    summary_parts.append("\n" + "-" * 50)
    summary_parts.append("TOP 5 PROTECTIVE FACTORS (Decrease Likelihood)")
    summary_parts.append("-" * 50)

    if 'logit_odds_ratio' in results.columns and 'logit_significant' in results.columns:
        sig_results = results[results['logit_significant'] == True]
        top_protective = sig_results[sig_results['logit_odds_ratio'] < 1].nsmallest(5, 'logit_odds_ratio')

        for i, (_, row) in enumerate(top_protective.iterrows(), 1):
            or_val = row['logit_odds_ratio']
            pct = (1 - or_val) * 100
            feature = row['feature'] if 'feature' in row else row.name
            summary_parts.append(
                f"  {i}. {feature}: {pct:.0f}% less likely (OR={or_val:.2f}, p={row['logit_p_value']:.4f})"
            )

    # Key Takeaways
    summary_parts.append("\n" + "-" * 50)
    summary_parts.append("KEY TAKEAWAYS")
    summary_parts.append("-" * 50)

    if n_sig_both > 0:
        summary_parts.append(f"  - {n_sig_both} features show consistent effects across both models")
        summary_parts.append("  - These high-confidence features should be prioritized for intervention")

    if 'logit_odds_ratio' in results.columns:
        high_impact = results[
            (results.get('logit_significant', False) == True) &
            ((results['logit_odds_ratio'] >= 2.0) | (results['logit_odds_ratio'] <= 0.5))
        ]
        if len(high_impact) > 0:
            summary_parts.append(f"  - {len(high_impact)} features have HIGH practical impact (OR >= 2.0 or <= 0.5)")

    summary_parts.append("\n" + "=" * 70)

    return "\n".join(summary_parts)