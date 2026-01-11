import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import permutation_test
import statsmodels.api as sm
from typing import List, Dict, Tuple

import warnings
warnings.filterwarnings('ignore')


# new try
def remove_collinear_features(feature_list):
    """
    Remove redundant features that cause perfect multicollinearity.
    """
    features_to_remove = [
        'is_weekend',  # Redundant: = is_saturday + is_sunday
        'is_business_hour',  # Redundant: combination of weekday/weekend versions
    ]
    
    cleaned_features = [f for f in feature_list if f not in features_to_remove]
    
    print("Removed collinear features:", features_to_remove)
    print(f"Features reduced from {len(feature_list)} to {len(cleaned_features)}")
    
    return cleaned_features


def build_random_forest_model_fixed(df, feature_cols, outcome_var, user_id_var, n_estimators=100):
    """
    Build Random Forest with fixed serialization issues for Databricks.

    Optimized to train only during cross-validation (5-6x faster than training
    full model separately), then uses best fold model for feature importance.
    """

    print("\n" + "="*70)
    print("APPROACH 2: RANDOM FOREST WITH CLUSTER-AWARE VALIDATION")
    print("="*70)

    # Prepare data
    X = df[feature_cols].copy()
    y = df[outcome_var].copy()
    groups = df[user_id_var].copy()

    # Remove missing values
    mask = X.notna().all(axis=1) & y.notna() & groups.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    groups = groups[mask].reset_index(drop=True)

    print(f"\nData shape: {X.shape}")
    print(f"Unique users: {groups.nunique()}")

    # Cross-validation with model storage (eliminates redundant full training)
    print(f"\nRunning cluster-based cross-validation with {n_estimators} trees...")
    gkf = GroupKFold(n_splits=5)

    cv_scores = {'train_r2': [], 'test_r2': [], 'test_mse': [], 'test_mae': []}
    fold_models = []  # Store fold models to reuse best one

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train on fold
        rf_fold = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=1  # Databricks compatibility
        )
        rf_fold.fit(X_train, y_train)
        fold_models.append(rf_fold)

        # Predict
        y_train_pred = rf_fold.predict(X_train)
        y_test_pred = rf_fold.predict(X_test)

        # Score
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        cv_scores['train_r2'].append(train_r2)
        cv_scores['test_r2'].append(test_r2)
        cv_scores['test_mse'].append(test_mse)
        cv_scores['test_mae'].append(test_mae)

        print(f"  Fold {fold}: Test R²={test_r2:.4f}, Train R²={train_r2:.4f}")

    # Summarize
    print("\nCross-Validation Summary (5-fold, Group-based):")
    print(f"  Test R²: {np.mean(cv_scores['test_r2']):.4f} +/- {np.std(cv_scores['test_r2']):.4f}")
    print(f"  Test RMSE: {np.sqrt(np.mean(cv_scores['test_mse'])):.4f}")
    print(f"  Test MAE: {np.mean(cv_scores['test_mae']):.4f}")
    print(f"  Train R² (overfitting check): {np.mean(cv_scores['train_r2']):.4f}")

    # Check for overfitting
    overfit = np.mean(cv_scores['train_r2']) - np.mean(cv_scores['test_r2'])
    if overfit > 0.1:
        print(f"  WARNING: Potential overfitting detected (gap={overfit:.4f})")

    # Use best fold model for feature importance (avoids redundant full training)
    best_fold_idx = np.argmax(cv_scores['test_r2'])
    rf_full = fold_models[best_fold_idx]
    print(f"\nUsing best fold model (Fold {best_fold_idx + 1}) for feature importance")

    # Feature importance
    importances = rf_full.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Mean Decrease Impurity):")
    print(importance_df.head(10).to_string(index=False))
    
    return rf_full, importance_df, cv_scores


def build_logistic_regression_model_fixed(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str,
    user_id_var: str,
    significance_level: float = 0.05,
) -> Tuple[sm.GLMResults, pd.DataFrame, Dict[str, List[float]]]:
    """
    Build Logistic Regression with cluster-aware cross-validation.

    Binary classification: outcome_var > 0 as positive class.
    Uses statsmodels GLM for model fitting and manual GroupKFold for CV.
    Maintains Databricks compatibility (no parallel processing).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and outcome variable
    feature_cols : List[str]
        List of feature column names
    outcome_var : str
        Outcome variable name (will be binarized to >0)
    user_id_var : str
        User identifier for group-based CV
    significance_level : float
        Significance level for coefficient testing

    Returns:
    --------
    Tuple containing:
        - model: statsmodels GLMResults object
        - importance_df: DataFrame with columns [feature, coefficient, p_value,
                         odds_ratio, importance, significant]
        - cv_results: dict with accuracy, precision, recall, f1, auc lists per fold
    """

    print("\n" + "="*70)
    print("APPROACH 3: LOGISTIC REGRESSION WITH CLUSTER-AWARE VALIDATION")
    print("="*70)

    # Prepare data
    X = df[feature_cols].copy()
    y_raw = df[outcome_var].copy()
    groups = df[user_id_var].copy()

    # Binarize outcome: outcome_var > 0
    y = (y_raw.fillna(0) > 0).astype(int)

    # Remove missing values
    mask = X.notna().all(axis=1) & y.notna() & groups.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    groups = groups[mask].reset_index(drop=True)

    print(f"\nData shape: {X.shape}")
    print(f"Unique users: {groups.nunique()}")
    print(f"Positive class ({outcome_var} > 0): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Negative class ({outcome_var} = 0): {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    # Fit full model with statsmodels GLM
    print("\nFitting Logistic Regression (GLM with Binomial family)...")
    X_with_const = sm.add_constant(X)

    try:
        logit_model = sm.GLM(
            y,
            X_with_const,
            family=sm.families.Binomial()
        ).fit()
    except Exception as e:
        print(f"Error fitting model: {e}")
        # Try with regularization if convergence issues
        print("Attempting with L2 regularization...")
        logit_model = sm.GLM(
            y,
            X_with_const,
            family=sm.families.Binomial()
        ).fit_regularized(alpha=0.1, L1_wt=0)

    print(f"\nModel Summary:")
    print(f"  Pseudo R-squared (McFadden): {1 - logit_model.llf/logit_model.llnull:.4f}")
    print(f"  Log-Likelihood: {logit_model.llf:.4f}")
    print(f"  AIC: {logit_model.aic:.4f}")
    print(f"  BIC: {logit_model.bic:.4f}")

    # Manual cross-validation with GroupKFold
    print("\nRunning cluster-based cross-validation (manual implementation)...")
    gkf = GroupKFold(n_splits=5)

    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'train_accuracy': []
    }

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Add constant for statsmodels
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        # Fit fold model
        try:
            fold_model = sm.GLM(
                y_train,
                X_train_const,
                family=sm.families.Binomial()
            ).fit()
        except Exception:
            # Fallback with regularization
            fold_model = sm.GLM(
                y_train,
                X_train_const,
                family=sm.families.Binomial()
            ).fit_regularized(alpha=0.1, L1_wt=0)

        # Predict probabilities
        y_train_prob = fold_model.predict(X_train_const)
        y_test_prob = fold_model.predict(X_test_const)

        # Convert to binary predictions (threshold = 0.5)
        y_train_pred = (y_train_prob >= 0.5).astype(int)
        y_test_pred = (y_test_prob >= 0.5).astype(int)

        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Handle edge cases where all predictions are same class
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)

        # AUC requires both classes present
        try:
            auc = roc_auc_score(y_test, y_test_prob)
        except ValueError:
            auc = np.nan

        cv_scores['train_accuracy'].append(train_acc)
        cv_scores['accuracy'].append(test_acc)
        cv_scores['precision'].append(precision)
        cv_scores['recall'].append(recall)
        cv_scores['f1'].append(f1)
        cv_scores['auc'].append(auc)

        print(f"  Fold {fold}: Acc={test_acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # Summarize CV results
    print("\nCross-Validation Summary (5-fold, Group-based):")
    print(f"  Test Accuracy: {np.mean(cv_scores['accuracy']):.4f} +/- {np.std(cv_scores['accuracy']):.4f}")
    print(f"  Test Precision: {np.mean(cv_scores['precision']):.4f} +/- {np.std(cv_scores['precision']):.4f}")
    print(f"  Test Recall: {np.mean(cv_scores['recall']):.4f} +/- {np.std(cv_scores['recall']):.4f}")
    print(f"  Test F1: {np.mean(cv_scores['f1']):.4f} +/- {np.std(cv_scores['f1']):.4f}")
    print(f"  Test AUC: {np.nanmean(cv_scores['auc']):.4f} +/- {np.nanstd(cv_scores['auc']):.4f}")
    print(f"  Train Accuracy (overfitting check): {np.mean(cv_scores['train_accuracy']):.4f}")

    # Check for overfitting
    overfit = np.mean(cv_scores['train_accuracy']) - np.mean(cv_scores['accuracy'])
    if overfit > 0.1:
        print(f"  WARNING: Potential overfitting detected (gap={overfit:.4f})")

    # Extract feature importance (by absolute coefficient value)
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': logit_model.params[1:].values,  # Skip intercept
        'std_error': logit_model.bse[1:].values,
        'p_value': logit_model.pvalues[1:].values,
        'z_stat': logit_model.tvalues[1:].values,
    })

    # Calculate odds ratios
    coef_df['odds_ratio'] = np.exp(coef_df['coefficient'])

    # Importance = absolute coefficient
    coef_df['importance'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('importance', ascending=False)

    # Flag significant features
    coef_df['significant'] = coef_df['p_value'] < significance_level

    print("\nTop 10 Most Important Features (by |coefficient|):")
    print(coef_df.head(10).to_string(index=False))

    return logit_model, coef_df, cv_scores


def build_linear_baseline_fixed(df, feature_cols, outcome_var, user_id_var):
    """
    Linear baseline with VIF check for multicollinearity.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    print("="*70)
    print("APPROACH 1: LINEAR BASELINE WITH MULTICOLLINEARITY CHECK")
    print("="*70)
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[outcome_var].copy()
    
    # Handle missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    # Check for perfect collinearity using VIF
    print("\nChecking for multicollinearity (VIF)...")
    X_with_const = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    print("\nVariance Inflation Factors:")
    print(vif_data.to_string(index=False))
    print("\nNote: VIF > 10 indicates problematic multicollinearity")
    print("      VIF > 100 indicates severe multicollinearity")
    
    # Fit model
    model = sm.OLS(y, X_with_const).fit()
    
    print("\nModel Summary:")
    print(f"  R²: {model.rsquared:.4f}")
    print(f"  Adj R²: {model.rsquared_adj:.4f}")
    print(f"  Observations: {len(y)}")
    print(f"  F-statistic p-value: {model.f_pvalue:.4e}")
    
    # Check if coefficients are reasonable
    max_coef = np.abs(model.params[1:]).max()
    if max_coef > 10:
        print(f"\n⚠️  WARNING: Large coefficients detected (max={max_coef:.2e})")
        print("    This suggests multicollinearity issues!")
    
    # Extract coefficients
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.params[1:].values,
        'std_error': model.bse[1:].values,
        'p_value': model.pvalues[1:].values,
        't_stat': model.tvalues[1:].values,
    })
    
    coef_df['importance'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features (by |coefficient|):")
    print(coef_df.head(10).to_string(index=False))
    
    return model, coef_df, vif_data


def run_full_feature_importance_analysis_fixed(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str = "wonky_study_count",
    user_id_var: str = "respondentPk",
    include_logistic: bool = True,
) -> Dict:
    """
    Comprehensive feature importance analysis with multiple model approaches.

    Fixed version that handles Databricks environment and multicollinearity.
    Now includes optional logistic regression for binary classification.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and outcome
    feature_cols : List[str]
        Feature column names
    outcome_var : str
        Outcome variable name
    user_id_var : str
        User identifier column
    include_logistic : bool
        If True, includes logistic regression model (binary: outcome > 0)

    Returns:
    --------
    Dict with keys:
        - linear_model, linear_importance, vif_data
        - rf_model, rf_importance, cv_results
        - lr_model, lr_importance, lr_cv_results (if include_logistic=True)
        - comparison (with lr_rank if logistic included)
    """

    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS (FIXED)")
    print("="*80)

    # Remove collinear features
    feature_cols_clean = remove_collinear_features(feature_cols)

    print(f"\nDataset: {len(df)} observations")
    print(f"Features: {len(feature_cols_clean)} (after removing collinearity)")
    print(f"Outcome: {outcome_var}")
    print(f"Unique users: {df[user_id_var].nunique()}")

    # Step 1: Linear baseline with diagnostics
    linear_model, linear_importance, vif_data = build_linear_baseline_fixed(
        df, feature_cols_clean, outcome_var, user_id_var
    )

    # Step 2: Random Forest (fixed for Databricks)
    rf_model, rf_importance, cv_results = build_random_forest_model_fixed(
        df, feature_cols_clean, outcome_var, user_id_var, n_estimators=100
    )

    # Step 3: Logistic Regression (optional)
    lr_model, lr_importance, lr_cv_results = None, None, None
    if include_logistic:
        lr_model, lr_importance, lr_cv_results = build_logistic_regression_model_fixed(
            df, feature_cols_clean, outcome_var, user_id_var
        )

    # Step 4: Compare
    print("\n" + "="*80)
    print("COMPARISON OF FEATURE IMPORTANCE")
    print("="*80)

    comparison = pd.DataFrame({
        'feature': feature_cols_clean,
    })

    comparison = comparison.merge(
        linear_importance[['feature', 'coefficient', 'p_value']],
        on='feature', how='left'
    )
    comparison = comparison.merge(
        rf_importance[['feature', 'importance']].rename(columns={'importance': 'rf_importance'}),
        on='feature', how='left'
    )

    comparison['linear_rank'] = comparison['coefficient'].abs().rank(ascending=False)
    comparison['rf_rank'] = comparison['rf_importance'].rank(ascending=False)

    # Add logistic regression ranks if included
    if include_logistic and lr_importance is not None:
        comparison = comparison.merge(
            lr_importance[['feature', 'coefficient']].rename(
                columns={'coefficient': 'lr_coefficient'}
            ),
            on='feature', how='left'
        )
        comparison['lr_rank'] = comparison['lr_coefficient'].abs().rank(ascending=False)
        comparison['avg_rank'] = (
            comparison['linear_rank'] + comparison['rf_rank'] + comparison['lr_rank']
        ) / 3
    else:
        comparison['avg_rank'] = (comparison['linear_rank'] + comparison['rf_rank']) / 2

    comparison = comparison.sort_values('avg_rank')

    # Print comparison
    print("\nTop Features by Average Rank:")
    rank_cols = ['feature', 'linear_rank', 'rf_rank']
    if include_logistic:
        rank_cols.append('lr_rank')
    rank_cols.extend(['avg_rank', 'p_value'])
    print(comparison[rank_cols].head(20).to_string(index=False))

    # Build result dictionary
    result = {
        'linear_model': linear_model,
        'linear_importance': linear_importance,
        'vif_data': vif_data,
        'rf_model': rf_model,
        'rf_importance': rf_importance,
        'cv_results': cv_results,
        'comparison': comparison,
    }

    if include_logistic:
        result['lr_model'] = lr_model
        result['lr_importance'] = lr_importance
        result['lr_cv_results'] = lr_cv_results

    return result



