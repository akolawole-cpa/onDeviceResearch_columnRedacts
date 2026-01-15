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