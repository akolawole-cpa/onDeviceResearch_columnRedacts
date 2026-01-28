"""
Modelling Utilities - Simplified

VIF analysis and helper functions for feature importance modelling.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore")


def calculate_vif(
    user_info_df: pd.DataFrame,
    test_results_df: pd.DataFrame,
    feature_cols: List[str],
    threshold_high: float = 10.0,
    threshold_moderate: float = 5.0,
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for multicollinearity detection.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing features
    feature_cols : List[str]
        List of feature column names
    threshold_high : float
        VIF threshold for high multicollinearity (default 10.0)
    threshold_moderate : float
        VIF threshold for moderate multicollinearity (default 5.0)

    Returns:
    --------
    pd.DataFrame with columns: feature, VIF, multicollinearity_flag
    """
    X = user_info_df[feature_cols].dropna()

    if len(X) == 0:
        print("⚠ No complete cases for VIF calculation")
        return pd.DataFrame(
            {
                "feature": feature_cols,
                "VIF": np.nan,
                "multicollinearity_flag": "insufficient_data",
            }
        )

    # Add constant for VIF calculation
    X_with_const = X.copy()
    X_with_const["_const"] = 1

    vif_data = []
    for i, col in enumerate(feature_cols):
        try:
            col_idx = list(X_with_const.columns).index(col)
            vif = variance_inflation_factor(X_with_const.values, col_idx)
            vif_data.append({"feature": col, "VIF": vif})
        except Exception as e:
            vif_data.append({"feature": col, "VIF": np.nan})

    vif_df = pd.DataFrame(vif_data)

    def flag_vif(x):
        if pd.isna(x):
            return "error"
        elif x > threshold_high:
            return "high"
        elif x > threshold_moderate:
            return "moderate"
        else:
            return "low"

    vif_df["multicollinearity_flag"] = vif_df["VIF"].apply(flag_vif)

    # Summary
    high_vif = vif_df[vif_df["multicollinearity_flag"] == "high"]
    moderate_vif = vif_df[vif_df["multicollinearity_flag"] == "moderate"]
    low_vif = vif_df[vif_df["multicollinearity_flag"] == "low"]

    print(f"VIF Summary:")
    print(f"   🔴 High (>{threshold_high}): {len(high_vif)} features")
    print(
        f"   🟡 Moderate ({threshold_moderate}-{threshold_high}): {len(moderate_vif)} features"
    )
    print(f"   🟢 Low (<{threshold_moderate}): {len(low_vif)} features")

    if len(high_vif) > 0:
        print(f"\n⚠ High VIF features:")
        for _, row in high_vif.iterrows():
            print(f"   {row['feature']}: {row['VIF']:.2f}")

    feature_to_set = dict(
        zip(test_results_df["feature"], test_results_df["feature_set"])
    )

    vif_df["feature_set"] = vif_df["feature"].map(feature_to_set)
    vif_df = vif_df[["feature_set", "feature", "VIF", "multicollinearity_flag"]]

    return vif_df.sort_values("VIF", ascending=False)


def extract_stat_coefficients(
    test_results_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Extract OLS coefficients from test_results_df for specified features.

    Parameters:
    -----------
    test_results_df : pd.DataFrame
        DataFrame with test results containing 'feature', 'ols_coefficient',
        'ols_p_value', 'ols_significant', 'cohens_d', 'cohens_d_magnitude'
    feature_cols : List[str]
        List of features to extract coefficients for

    Returns:
    --------
    pd.DataFrame with columns: feature, feature_set, ols_coefficient, ols_pvalue,
                               ols_significant, cohens_d, cohens_d_magnitude
    """
    # Filter to requested features
    stat_df = test_results_df[test_results_df["feature"].isin(feature_cols)].copy()

    # Select relevant columns (using current column names)
    result = stat_df[
        [
            "feature",
            "feature_set",
            "ols_coefficient",
            "ols_p_value",
            "ols_significant",
            "ols_interpretation",
            "ols_interpretation_short",
            "cohens_d",
            "cohens_d_magnitude",
            "odds_ratio",
            "logit_p_value",
            "ols_significant",
            "lr_interpretation",
            "lr_interpretation_short",
            "significant_both"
        ]
    ].copy()

    # Rename for consistency
    result = result.rename(
        columns={
            "ols_p_value": "ols_pvalue",
        }
    )

    # Add absolute coefficient for ranking
    result["ols_abs_coefficient"] = result["ols_coefficient"].abs()

    # Report missing features
    found_features = set(result["feature"].tolist())
    missing = [f for f in feature_cols if f not in found_features]
    if missing:
        print(f"⚠ {len(missing)} features not found in test_results_df")

    print(f"✓ Extracted statistical coefficients for {len(result)} features")

    return result.sort_values("ols_abs_coefficient", ascending=False)
