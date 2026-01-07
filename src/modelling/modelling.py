import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import permutation_test
import statsmodels.api as sm

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
    
    # Train model with n_jobs=1 to avoid serialization issues
    rf_full = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,  # Reduced to avoid overfitting on small dataset
        min_samples_split=20,  # Increased for small dataset
        min_samples_leaf=10,
        random_state=42,
        n_jobs=1,  # FIX: Disable parallelization for Databricks
        verbose=0
    )
    
    print(f"\nTraining Random Forest with {n_estimators} trees...")
    rf_full.fit(X, y)
    
    # Manual cross-validation to avoid joblib serialization
    print("Running cluster-based cross-validation (manual implementation)...")
    gkf = GroupKFold(n_splits=5)
    
    cv_scores = {'train_r2': [], 'test_r2': [], 'test_mse': [], 'test_mae': []}
    
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
            n_jobs=1
        )
        rf_fold.fit(X_train, y_train)
        
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
    print(f"  Test R²: {np.mean(cv_scores['test_r2']):.4f} ± {np.std(cv_scores['test_r2']):.4f}")
    print(f"  Test RMSE: {np.sqrt(np.mean(cv_scores['test_mse'])):.4f}")
    print(f"  Test MAE: {np.mean(cv_scores['test_mae']):.4f}")
    print(f"  Train R² (overfitting check): {np.mean(cv_scores['train_r2']):.4f}")
    
    # Check for overfitting
    overfit = np.mean(cv_scores['train_r2']) - np.mean(cv_scores['test_r2'])
    if overfit > 0.1:
        print(f"  ⚠️  WARNING: Potential overfitting detected (gap={overfit:.4f})")
    
    # Feature importance
    importances = rf_full.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Mean Decrease Impurity):")
    print(importance_df.head(10).to_string(index=False))
    
    return rf_full, importance_df, cv_scores


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
    df, 
    feature_cols, 
    outcome_var="wonky_study_count",
    user_id_var="respondentPk"
):
    """
    Fixed version that handles Databricks environment and multicollinearity.
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
    
    # Step 3: Compare
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
    comparison['avg_rank'] = (comparison['linear_rank'] + comparison['rf_rank']) / 2
    
    comparison = comparison.sort_values('avg_rank')
    
    print("\nTop Features by Average Rank:")
    print(comparison[['feature', 'linear_rank', 'rf_rank', 'avg_rank', 'p_value']].to_string(index=False))
    
    return {
        'linear_model': linear_model,
        'linear_importance': linear_importance,
        'vif_data': vif_data,
        'rf_model': rf_model,
        'rf_importance': rf_importance,
        'cv_results': cv_results,
        'comparison': comparison,
    }



