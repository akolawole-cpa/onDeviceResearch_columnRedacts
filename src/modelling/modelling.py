"""
Modelling Module - Random Forest with SHAP Analysis

Core model building functions focused on:
- Random Forest with GroupKFold CV
- SHAP feature importance and interactions
- Optional grid search for hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import List, Dict, Tuple, Any, Optional
import shap
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# RANDOM FOREST MODEL
# =============================================================================


def build_random_forest(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str,
    user_id_var: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    max_features: Optional[Any] = None,
    n_splits: int = 5,
    do_gridsearch: bool = False,
    param_grid: Optional[Dict] = None,
) -> Tuple[Any, pd.DataFrame, Dict, pd.DataFrame]:
    """
    Build Random Forest with cluster-aware cross-validation.

    Returns:
    --------
    Tuple of (model, X_data, cv_metrics, importance_df)
    """
    print("=" * 60)
    print("RANDOM FOREST MODEL")
    print("=" * 60)

    # Prepare data
    X = df[feature_cols].copy()
    y = df[outcome_var].copy()
    groups = df[user_id_var].copy()

    mask = X.notna().all(axis=1) & y.notna() & groups.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    groups = groups[mask].reset_index(drop=True)

    print(f"\nObservations: {len(y):,}")
    print(f"Unique users: {groups.nunique():,}")
    print(f"Features: {len(feature_cols)}")

    # Grid search (optional)
    if do_gridsearch:
        print("\n🔍 Running Grid Search...")
        if param_grid is None:
            print("SET THE PARAM_GRID!!!")

        base_rf = RandomForestRegressor(
            min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1
        )
        gkf = GroupKFold(n_splits=n_splits)
        cv_splits = list(gkf.split(X, y, groups))

        grid_search = GridSearchCV(
            base_rf, param_grid, cv=cv_splits, scoring="r2", n_jobs=1, verbose=1
        )
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        print(f"✓ Best params: {best_params}")
        print(f"  Best CV R²: {grid_search.best_score_:.4f}")

        n_estimators = best_params.get("n_estimators", n_estimators)
        max_depth = best_params.get("max_depth", max_depth)
        max_features = best_params.get("max_features", max_features)
    else:
        print(
            f"Using default params: n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}"
        )

    # Cross-validation
    print(f"\nRunning {n_splits}-fold CV...")
    gkf = GroupKFold(n_splits=n_splits)

    cv_metrics = {"train_r2": [], "test_r2": [], "test_rmse": [], "test_mae": []}
    fold_models = []

    rf_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if max_features:
        rf_params["max_features"] = max_features

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        fold_models.append(rf)

        train_r2 = r2_score(y_train, rf.predict(X_train))
        test_r2 = r2_score(y_test, rf.predict(X_test))
        test_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
        test_mae = mean_absolute_error(y_test, rf.predict(X_test))

        cv_metrics["train_r2"].append(train_r2)
        cv_metrics["test_r2"].append(test_r2)
        cv_metrics["test_rmse"].append(test_rmse)
        cv_metrics["test_mae"].append(test_mae)

        print(f"  Fold {fold}: Test R²={test_r2:.4f}, Train R²={train_r2:.4f}")
        print(f"  Fold {fold}: Test RMSE={test_rmse:.4f}, Test MAE={test_mae:.4f}")

    # Summary
    print(f"\nCV Summary:")
    print(
        f"  Test R²:   {np.mean(cv_metrics['test_r2']):.4f} ± {np.std(cv_metrics['test_r2']):.4f}"
    )
    print(f"  Test RMSE: {np.mean(cv_metrics['test_rmse']):.4f}")
    print(f"  Test MAE:  {np.mean(cv_metrics['test_mae']):.4f}")

    overfit_gap = np.mean(cv_metrics["train_r2"]) - np.mean(cv_metrics["test_r2"])
    status = "Possible overfitting" if overfit_gap > 0.1 else "OK"
    print(f"  Overfit gap: {overfit_gap:.4f} ({status})")

    # Select best fold model
    best_idx = np.argmax(cv_metrics["test_r2"])
    best_model = fold_models[best_idx]
    print(f"\n✓ Using best fold model (fold {best_idx + 1})")

    # Feature importance (MDI)
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "rf_importance": best_model.feature_importances_,
        }
    )
    importance_df["rf_importance_pct"] = importance_df["rf_importance"] * 100
    importance_df = importance_df.sort_values("rf_importance", ascending=False)

    return best_model, X, cv_metrics, importance_df


# =============================================================================
# SHAP ANALYSIS
# =============================================================================


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    sample_size: int = 1000,
) -> Tuple[shap.Explainer, np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for feature importance interpretation.

    Returns:
    --------
    Tuple of (explainer, shap_values, shap_importance_df)
    """
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)

    # Sample if large dataset
    if len(X) > sample_size:
        print(f"Sampling {sample_size:,} observations for SHAP...")
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    print(f"Computing SHAP values for {len(X_sample):,} observations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP importance
    shap_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "shap_importance": np.abs(shap_values).mean(axis=0),
        }
    )
    shap_importance["shap_importance_pct"] = (
        shap_importance["shap_importance"]
        / shap_importance["shap_importance"].sum()
        * 100
    )
    shap_importance = shap_importance.sort_values("shap_importance", ascending=False)

    print(f"\n✓ SHAP values computed")
    print(f"\nTop 10 features by SHAP importance:")
    print(shap_importance.head(10).to_string(index=False))

    return explainer, shap_values, shap_importance


def compute_shap_interactions(
    explainer: shap.Explainer,
    X: pd.DataFrame,
    sample_size: int = 250,
    top_n: int = 20,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP interaction values to identify feature interactions.

    Returns:
    --------
    Tuple of (interaction_values, interaction_summary_df)
    """
    print("\n" + "-" * 40)
    print("SHAP INTERACTIONS")
    print("-" * 40)

    # Sample - interactions are computationally expensive
    if len(X) > sample_size:
        print(f"Sampling {sample_size:,} observations for interactions...")
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    print(f"Computing interaction values (this may take a while)...")
    interaction_values = explainer.shap_interaction_values(X_sample)

    # Summarize interactions
    n_features = len(X.columns)
    interaction_strength = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Mean absolute interaction effect
            strength = np.abs(interaction_values[:, i, j]).mean()
            interaction_strength.append(
                {
                    "feature_1": X.columns[i],
                    "feature_2": X.columns[j],
                    "interaction_strength": strength,
                }
            )

    interaction_df = pd.DataFrame(interaction_strength)
    interaction_df = interaction_df.sort_values("interaction_strength", ascending=False)
    interaction_df["rank"] = range(1, len(interaction_df) + 1)

    print(f"\n✓ Computed {len(interaction_df):,} pairwise interactions")
    print(f"\nTop {top_n} feature interactions:")
    print(interaction_df.head(top_n).to_string(index=False))

    return interaction_values, interaction_df


def get_shap_contributions(
    shap_values: np.ndarray,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean SHAP contribution (signed) per feature.

    This shows the average directional effect on the outcome:
    - Positive: feature increases wonkiness
    - Negative: feature decreases wonkiness

    Returns:
    --------
    pd.DataFrame with feature, mean_contribution, abs_contribution
    """
    contributions = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_contribution": shap_values.mean(axis=0),
            "abs_contribution": np.abs(shap_values).mean(axis=0),
        }
    )

    contributions["variance_scale"] = ((
        contributions["abs_contribution"] - contributions["mean_contribution"].abs()
    ) / contributions["abs_contribution"]).round(2)

    contributions["direction"] = contributions["mean_contribution"].apply(
        lambda x: "↑ increases" if x > 0 else "↓ decreases"
    )
    return contributions.sort_values("abs_contribution", ascending=False)


# =============================================================================
# COMBINED RESULTS
# =============================================================================


def create_feature_summary(
    rf_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    shap_contributions: pd.DataFrame,
    stats_coefficients: pd.DataFrame,
    vif_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create unified feature summary table combining all metrics.

    Parameters:
    -----------
    rf_importance : pd.DataFrame
        From build_random_forest, contains rf_importance, rf_importance_pct
    shap_importance : pd.DataFrame
        From compute_shap_values, contains shap_importance, shap_importance_pct
    shap_contributions : pd.DataFrame
        From get_shap_contributions, contains mean_contribution, direction
    stats_coefficients : pd.DataFrame
        From extract_ols_coefficients, contains ols_coefficient, ols_pvalue, etc.
    vif_data : pd.DataFrame
        From calculate_vif, contains VIF, multicollinearity_flag

    Returns:
    --------
    pd.DataFrame - unified feature summary sorted by SHAP importance
    """
    summary = rf_importance[["feature", "rf_importance", "rf_importance_pct"]].copy()

    # Merge SHAP importance
    summary = summary.merge(
        shap_importance[["feature", "shap_importance", "shap_importance_pct"]],
        on="feature",
        how="left",
    )

    # Merge SHAP contributions
    summary = summary.merge(
        shap_contributions[["feature", "mean_contribution", "direction"]],
        on="feature",
        how="left",
    )

    # Merge OLS coefficients (from test_results_df)
    stats_cols = [
        "feature",
        "ols_coefficient",
        "ols_pvalue",
        "significant_both",
        "ols_cohens_d",
        "ols_effect_size",
        'ols_interpretation_short',
        'odds_ratio',
        'logit_p_value',
        'lr_interpretation_short',
    ]
    available_cols = [c for c in stats_cols if c in stats_coefficients.columns]
    summary = summary.merge(stats_coefficients[available_cols], on="feature", how="left")

    # Merge VIF
    summary = summary.merge(
        vif_data[["feature", "VIF", "multicollinearity_flag"]], on="feature", how="left"
    )

    # Formating
    summary["shap_importance_pct"] = summary["shap_importance_pct"].round(3)/100
    summary["rf_importance_pct"] = summary["rf_importance_pct"].round(3)/100

    # Rank by SHAP importance
    summary = summary.sort_values("shap_importance", ascending=False)
    summary["rank"] = range(1, len(summary) + 1)

    return summary
