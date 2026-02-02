"""
Statistical Tests Module - Simplified & Streamlined

Runs OLS and Logistic Regression with cluster-robust standard errors.
Outputs a single combined results table with interpretations.

Main executing function: run_statistical_tests()

Output columns:
- feature: Feature name
- feature_type: 'binary' or 'numeric'
- n_wonky, n_non_wonky: Sample sizes for each group
- wonky_mean, non_wonky_mean: Outcome means by group
- ols_coefficient, ols_se, ols_t_stat, ols_p_value, ols_significant
- cohens_d, cohens_d_magnitude
- ols_interpretation: Type-aware interpretation
- odds_ratio, or_ci_lower, or_ci_upper
- logit_coefficient, logit_se, logit_z_stat, logit_p_value, logit_significant  
- or_magnitude
- lr_interpretation: Type-aware interpretation
- significant_both, significant_either
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Logit

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("joblib not installed. Parallel processing disabled.")


# INTERNAL HELPERS

def _get_significance_stars(p: float) -> str:
    """Return significance stars based on p-value."""
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size magnitude."""
    if pd.isna(d):
        return "undefined"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    return "large"


def _interpret_odds_ratio(or_val: float) -> str:
    """Interpret odds ratio effect size magnitude."""
    if pd.isna(or_val) or or_val <= 0:
        return "undefined"
    # Use max(OR, 1/OR) for symmetric interpretation
    or_effect = max(or_val, 1/or_val)
    if or_effect < 1.5:
        return "negligible"
    elif or_effect < 2.5:
        return "small"
    elif or_effect < 4.0:
        return "medium"
    return "large"


def _detect_feature_type(series: pd.Series) -> str:
    """
    Detect if a feature is binary or numeric.
    
    Binary: only contains values in {0, 1} (or {0}, {1}, {True, False})
    Numeric: everything else (continuous, ordinal with >2 values, etc.)
    """
    unique_vals = series.dropna().unique()
    
    # Check if binary (0/1 or True/False)
    if len(unique_vals) <= 2:
        unique_set = set(unique_vals)
        if unique_set <= {0, 1, True, False, 0.0, 1.0}:
            return 'binary'
    
    return 'numeric'


# SINGLE FEATURE FITTING (for parallel execution)

def _fit_single_feature(
    feature: str,
    df: pd.DataFrame,
    outcome_var: str,
    outcome_binary: str,
    user_id_var: str,
    significance_level: float,
    baseline_mean: float,
    min_threshold: int = 5,
) -> Optional[Dict]:
    """
    Fit both OLS and Logistic regression for a single feature.
    Returns combined results dictionary.
    
    Note on naming convention:
    - wonky_mean: mean of the FEATURE when outcome_var > 0 (i.e., among wonky respondents)
    - non_wonky_mean: mean of the FEATURE when outcome_var == 0 (i.e., among non-wonky respondents)
    - n_wonky: count of wonky respondents (outcome > 0)
    - n_non_wonky: count of non-wonky respondents (outcome == 0)
    """
    if feature not in df.columns:
        return None
    
    cols_needed = [user_id_var, outcome_var, outcome_binary, feature]
    df_test = df[cols_needed].dropna()
    
    # Checks
    if len(df_test) < 30 or df_test[feature].nunique() < 2:
        print("needs variation, check sample and target feature")
        return None
    if df_test[outcome_binary].nunique() < 2:
        print("needs variation, check target feature")
        return None
    
    feature_type = _detect_feature_type(df_test[feature])
    
    mask_wonky = df_test[outcome_binary] > 0  
    mask_non_wonky = df_test[outcome_binary] == 0  
    
    n_wonky = mask_wonky.sum()
    n_non_wonky = mask_non_wonky.sum()
    
    if n_wonky < min_threshold or n_non_wonky < min_threshold:
        return None
    
    wonky_mean = df_test.loc[mask_wonky, feature].mean()
    non_wonky_mean = df_test.loc[mask_non_wonky, feature].mean()
    
    if feature_type == 'binary':
        mask_with_feature = df_test[feature] == 1
        mask_without_feature = df_test[feature] == 0
        n_with_feature = mask_with_feature.sum()
        n_without_feature = mask_without_feature.sum()
        
        if n_with_feature < min_threshold and n_without_feature < min_threshold:
            return None
        
        if n_with_feature >= min_threshold:
            mean_outcome_with_feature = df_test.loc[mask_with_feature, outcome_var].mean()
        else:
            mean_outcome_with_feature = None
            
        if n_without_feature >= min_threshold:
            mean_outcome_without_feature = df_test.loc[mask_without_feature, outcome_var].mean()
        else:
            mean_outcome_without_feature = None
    else:
        n_with_feature = None
        n_without_feature = None
        mean_outcome_with_feature = None
        mean_outcome_without_feature = None
    
    result = {
        'feature': feature,
        'feature_type': feature_type,
        # Sample sizes by outcome group
        'n_wonky': int(n_wonky),
        'n_non_wonky': int(n_non_wonky),
        # Feature prevalence/mean by outcome group
        'wonky_mean': wonky_mean,
        'non_wonky_mean': non_wonky_mean,
        # Sample sizes by feature presence (binary only)
        'n_with_feature': int(n_with_feature) if n_with_feature is not None else None,
        'n_without_feature': int(n_without_feature) if n_without_feature is not None else None,
        # Outcome means by feature presence (binary only)
        'mean_outcome_with_feature': mean_outcome_with_feature,
        'mean_outcome_without_feature': mean_outcome_without_feature,
    }
    
    # --- OLS Regression ---
    try:
        X_ols = sm.add_constant(df_test[[feature]])
        y_ols = df_test[outcome_var]
        
        ols_model = OLS(y_ols, X_ols)
        ols_result = ols_model.fit(
            cov_type='cluster',
            cov_kwds={'groups': df_test[user_id_var]}
        )
        
        ols_coef = ols_result.params[feature]
        ols_se = ols_result.bse[feature]
        ols_p = ols_result.pvalues[feature]
        std_y = df_test[outcome_var].std()
        cohens_d = ols_coef / std_y if std_y > 0 else np.nan
        
        # OLS interpretation - type-aware
        if ols_p < significance_level:
            direction = "increases" if ols_coef > 0 else "decreases"
            
            if baseline_mean != 0:
                pct_change = (ols_coef / baseline_mean) * 100
                
                if feature_type == 'binary':
                    # Binary: "Having this feature increases wonkiness by X%"
                    ols_interpretation = f"{direction} wonkiness by {abs(pct_change):.2f}%"
                    ols_interpretation_short = f"{abs(pct_change):.2f}% {direction} "
                else:
                    # Numeric: "Each 1-unit increase raises wonkiness by X%"
                    ols_interpretation = f"unit increase {direction} wonkiness by {abs(pct_change):.2f}%"
                    ols_interpretation_short = f"{abs(pct_change):.2f}% {direction}"
            else:
                if feature_type == 'binary':
                    ols_interpretation = f"{direction} wonkiness by {abs(ols_coef):.3f} units"
                    ols_interpretation_short = f"{abs(ols_coef):.2f} {direction}"
                else:
                    ols_interpretation = f"unit increase {direction} wonkiness by {abs(ols_coef):.2f} units"
                    ols_interpretation_short = f"{abs(ols_coef):.2f} {direction}"
        else:
            ols_interpretation = "No significant effect"
            ols_interpretation_short = "-"
        
        result.update({
            'ols_coefficient': ols_coef,
            'ols_se': ols_se,
            'ols_t_stat': ols_result.tvalues[feature],
            'ols_p_value': ols_p,
            'ols_significant': ols_p < significance_level,
            'cohens_d': cohens_d,
            'cohens_d_magnitude': _interpret_cohens_d(cohens_d),
            'ols_interpretation': ols_interpretation,
            'ols_interpretation_short': ols_interpretation_short,
        })
    except Exception as e:
        result.update({
            'ols_coefficient': np.nan,
            'ols_se': np.nan,
            'ols_t_stat': np.nan,
            'ols_p_value': np.nan,
            'ols_significant': False,
            'cohens_d': np.nan,
            'cohens_d_magnitude': 'error',
            'ols_interpretation': f'OLS failed: {str(e)[:50]}',
            'ols_interpretation_short': f'OLS failed: {str(e)[:50]}',
        })
    
    # --- Logistic Regression ---
    try:
        X_logit = sm.add_constant(df_test[[feature]])
        y_logit = df_test[outcome_binary]
        
        logit_model = Logit(y_logit, X_logit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logit_result = logit_model.fit(
                cov_type='cluster',
                cov_kwds={'groups': df_test[user_id_var]},
                disp=False,
                maxiter=100,
            )
        
        logit_coef = logit_result.params[feature]
        logit_se = logit_result.bse[feature]
        logit_p = logit_result.pvalues[feature]
        odds_ratio = np.exp(logit_coef)
        
        # Confidence interval for OR
        conf_int = logit_result.conf_int().loc[feature]
        or_ci_lower = np.exp(conf_int[0])
        or_ci_upper = np.exp(conf_int[1])
        
        # LR interpretation - type-aware
        if logit_p < significance_level:
            if feature_type == 'binary':
                # Binary feature interpretations
                if odds_ratio > 5:
                    lr_interpretation = f"Very strong effect: odds {odds_ratio:.2f}x higher"
                    lr_interpretation_short = f"{odds_ratio:.2f}x higher"
                elif odds_ratio > 2:
                    lr_interpretation = f"Odds are {odds_ratio:.2f}x higher for wonky"
                    lr_interpretation_short = f"{odds_ratio:.2f}x higher"
                elif odds_ratio > 1:
                    pct = (odds_ratio - 1) * 100
                    lr_interpretation = f"{pct:.2f}% higher odds of being wonky"
                    lr_interpretation_short = f"{pct:.2f}% higher"
                elif odds_ratio < 0.2:
                    inverse_or = 1 / odds_ratio
                    lr_interpretation = f"Very strong protective: odds {inverse_or:.2f}x lower"
                    lr_interpretation_short = f"{inverse_or:.2f}x lower"
                elif odds_ratio < 0.5:
                    inverse_or = 1 / odds_ratio
                    lr_interpretation = f"Odds are {inverse_or:.2f}x lower for wonky"
                    lr_interpretation_short = f"{inverse_or:.2f}x lower"
                else:
                    pct = (1 - odds_ratio) * 100
                    lr_interpretation = f"{pct:.2f}% lower odds of being wonky"
                    lr_interpretation_short = f"{pct:.2f}% lower"
            else:
                # Numeric feature interpretations
                if odds_ratio > 2:
                    lr_interpretation = f"Each unit increase: odds {odds_ratio:.2f}x higher"
                    lr_interpretation_short = f"{odds_ratio:.2f}x higher"
                elif odds_ratio > 1:
                    pct = (odds_ratio - 1) * 100
                    lr_interpretation = f"Each unit increase: {pct:.2f}% higher odds"
                    lr_interpretation_short = f"{pct:.2f}% higher"
                elif odds_ratio < 0.5:
                    inverse_or = 1 / odds_ratio
                    lr_interpretation = f"Each unit increase: odds {inverse_or:.2f}x lower"
                    lr_interpretation_short = f"{inverse_or:.2f}x lower"
                else:
                    pct = (1 - odds_ratio) * 100
                    lr_interpretation = f"Each unit increase: {pct:.2f}% lower odds"
                    lr_interpretation_short = f"{pct:.2f}% lower"
        else:
            lr_interpretation = f"No significant effect (p={logit_p:.3f})"
        
        result.update({
            'odds_ratio': odds_ratio,
            'or_ci_lower': or_ci_lower,
            'or_ci_upper': or_ci_upper,
            'logit_coefficient': logit_coef,
            'logit_se': logit_se,
            'logit_z_stat': logit_result.tvalues[feature],
            'logit_p_value': logit_p,
            'logit_significant': logit_p < significance_level,
            'or_magnitude': _interpret_odds_ratio(odds_ratio),
            'lr_interpretation': lr_interpretation,
            'lr_interpretation_short': lr_interpretation_short,
        })
    except Exception as e:
        result.update({
            'odds_ratio': np.nan,
            'or_ci_lower': np.nan,
            'or_ci_upper': np.nan,
            'logit_coefficient': np.nan,
            'logit_se': np.nan,
            'logit_z_stat': np.nan,
            'logit_p_value': np.nan,
            'logit_significant': False,
            'or_magnitude': 'error',
            'lr_interpretation': f'Logit failed: {str(e)[:50]}',
            'lr_interpretation_short': f'Logit failed: {str(e)[:50]}',
        })
    
    return result


# MAIN FUNCTION

def run_statistical_tests(
    df: pd.DataFrame,
    feature_list: List[str],
    outcome_var: str = "wonky_study_count",
    user_id_var: str = "respondentPk",
    significance_level: float = 0.05,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run OLS and Logistic Regression tests for all features.
    
    This function runs both models in a single pass per feature:
    - OLS: For effect size estimates (coefficient), p-values, and Cohen's d
    - Logistic Regression: For odds ratios and alternative interpretation
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features, outcome variable, and user identifier
    feature_list : List[str]
        List of feature column names to test
    outcome_var : str
        Name of the outcome variable (count-based, e.g., wonky_study_count)
    user_id_var : str
        Name of the user identifier column for cluster-robust SE
    significance_level : float
        P-value threshold for significance (default: 0.05)
    n_jobs : int
        Number of parallel jobs. -1 = all cores, 1 = sequential
    verbose : bool
        Print progress information
    
    Returns
    -------
    pd.DataFrame
        Combined results with columns:
        - feature: Feature name
        - feature_type: 'binary' or 'numeric'
        - n_wonky, n_non_wonky: Sample sizes by outcome
        - wonky_mean, non_wonky_mean: Feature means by outcome group
        - n_with_feature, n_without_feature: Sample sizes (binary only)
        - ols_coefficient, ols_se, ols_t_stat, ols_p_value, ols_significant
        - cohens_d, cohens_d_magnitude
        - ols_interpretation: Type-aware interpretation
        - odds_ratio, or_ci_lower, or_ci_upper
        - logit_coefficient, logit_se, logit_z_stat, logit_p_value, logit_significant
        - or_magnitude
        - lr_interpretation: Type-aware interpretation
        - significant_both, significant_either
    
    Example
    -------
    >>> results = run_statistical_tests(
    ...     df=user_data,
    ...     feature_list=['is_weekend', 'days_active', 'quality_score'],
    ...     outcome_var='wonky_study_count',
    ...     user_id_var='respondentPk'
    ... )
    >>> results[['feature', 'feature_type', 'ols_interpretation', 'lr_interpretation']].head()
    """
    # Validate inputs
    valid_features = [f for f in feature_list if f in df.columns]
    missing = set(feature_list) - set(valid_features)
    
    if verbose and missing:
        print(f"⚠ {len(missing)} features not found in DataFrame:")
        for m in list(missing)[:5]:
            # Try to find similar column names
            similar = [c for c in df.columns if m[:20] in c or c[:20] in m]
            if similar:
                print(f"  '{m}' - similar columns found: {similar[:3]}")
            else:
                print(f"  '{m}' - no similar columns found")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    
    if not valid_features:
        print("No valid features to test.")
        return pd.DataFrame()
    
    if verbose:
        print(f"Testing {len(valid_features)} features...")
    
    # Prepare data
    cols = [user_id_var, outcome_var] + valid_features
    df_prep = df[cols].copy()
    df_prep[outcome_var] = df_prep[outcome_var].fillna(0)
    
    # Create binary outcome for logistic regression
    outcome_binary = f"{outcome_var}_binary"
    df_prep[outcome_binary] = (df_prep[outcome_var] > 0).astype(int)
    
    # Calculate baseline mean for interpretation
    baseline_mean = df_prep[outcome_var].mean()
    
    if verbose:
        print(f"Baseline mean ({outcome_var}): {baseline_mean:.4f}")
        print(f"Wonky rate: {df_prep[outcome_binary].mean():.2%}")
    
    # Run tests (parallel or sequential)
    if HAS_JOBLIB and n_jobs != 1 and len(valid_features) > 3:
        if verbose:
            print(f"Running in parallel (n_jobs={n_jobs})...")
        
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_single_feature)(
                f, df_prep, outcome_var, outcome_binary, 
                user_id_var, significance_level, baseline_mean
            )
            for f in valid_features
        )
    else:
        if verbose:
            print("Running sequentially...")
        
        results = [
            _fit_single_feature(
                f, df_prep, outcome_var, outcome_binary,
                user_id_var, significance_level, baseline_mean
            )
            for f in valid_features
        ]
    
    # Filter out None results and create DataFrame
    results = [r for r in results if r is not None]
    
    if not results:
        print("No features passed validation checks.")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Add combined significance columns
    results_df['significant_both'] = (
        results_df['ols_significant'] & results_df['logit_significant']
    )
    results_df['significant_either'] = (
        results_df['ols_significant'] | results_df['logit_significant']
    )
    
    # Add significance stars
    results_df['ols_stars'] = results_df['ols_p_value'].apply(_get_significance_stars)
    results_df['logit_stars'] = results_df['logit_p_value'].apply(_get_significance_stars)
    
    # Sort by OLS t-statistic magnitude
    results_df = results_df.sort_values('ols_t_stat', key=abs, ascending=False)
    results_df = results_df.set_index('feature')
    
    if verbose:
        n_sig_both = results_df['significant_both'].sum()
        n_sig_ols = results_df['ols_significant'].sum()
        n_sig_logit = results_df['logit_significant'].sum()
        n_binary = (results_df['feature_type'] == 'binary').sum()
        n_numeric = (results_df['feature_type'] == 'numeric').sum()
        print(f"\n✓ Testing complete!")
        print(f"  Feature types:        {n_binary} binary, {n_numeric} numeric")
        print(f"  Significant (OLS):    {n_sig_ols}/{len(results_df)}")
        print(f"  Significant (Logit):  {n_sig_logit}/{len(results_df)}")
        print(f"  Significant (Both):   {n_sig_both}/{len(results_df)}")
    
    return results_df


def get_summary_table(
    results_df: pd.DataFrame,
    top_n: int = 20,
    sort_by: str = 'ols_t_stat',
) -> pd.DataFrame:
    """
    Create a stakeholder-friendly summary table.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_statistical_tests()
    top_n : int
        Number of top features to include
    sort_by : str
        Column to sort by (default: 'ols_t_stat')
    
    Returns
    -------
    pd.DataFrame
        Summary with key columns for reporting
    """
    df = results_df.reset_index().copy()
    
    df = df.sort_values(sort_by, key=abs, ascending=False).head(top_n)
    
    summary_cols = {
        'feature': 'Feature',
        'feature_type': 'Type',
        'n_wonky': 'N (wonky)',
        'n_non_wonky': 'N (non-wonky)',
        'wonky_mean': 'Mean (wonky)',
        'non_wonky_mean': 'Mean (non-wonky)',
        'ols_coefficient': 'OLS Coef',
        'ols_p_value': 'OLS p-value',
        'ols_stars': '',
        'cohens_d': "Cohen's d",
        'ols_interpretation': 'OLS Interpretation',
        'odds_ratio': 'Odds Ratio',
        'or_ci_lower': 'OR CI Lower',
        'or_ci_upper': 'OR CI Upper',
        'logit_p_value': 'LR p-value',
        'logit_stars': '',
        'lr_interpretation': 'LR Interpretation',
        'significant_both': 'Sig. Both',
    }
    
    available = [c for c in summary_cols.keys() if c in df.columns]
    summary = df[available].copy()
    summary.columns = [summary_cols[c] for c in available]
    
    return summary