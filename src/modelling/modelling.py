"""
Modelling Module

Core model building functions for feature importance analysis.
Contains: Linear Regression, Logistic Regression, Random Forest

Usage:
    from modelling.modelling import (
        build_linear_baseline,
        build_logistic_regression_model,
        build_random_forest_model,
        build_ensemble_random_forest,
        run_all_models,
        create_user_level_train_test_split,
        evaluate_on_holdout,
        EnsembleRandomForest,
    )
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Dict, Tuple, Any, Optional

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# LINEAR REGRESSION
# =============================================================================

def build_linear_baseline(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str,
    user_id_var: str,
    calculate_vif: bool = True,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Build Linear OLS Regression with VIF diagnostics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and outcome
    feature_cols : List[str]
        List of feature column names
    outcome_var : str
        Target variable name
    user_id_var : str
        User identifier (for info only, not used in basic OLS)
    calculate_vif : bool
        Whether to calculate VIF (can be slow for many features)
    
    Returns:
    --------
    Tuple of (model, importance_df, vif_df)
        - model: fitted statsmodels OLS model
        - importance_df: DataFrame with coefficients, p-values, importance
        - vif_df: DataFrame with VIF for each feature
    """
    print("\n" + "="*70)
    print("LINEAR REGRESSION (OLS)")
    print("="*70)
    
    X = df[feature_cols].copy()
    y = df[outcome_var].copy()
    
    # Handle missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    print(f"\nObservations: {len(y)}")
    print(f"Features: {len(feature_cols)}")
    
    # VIF calculation (optional - can be slow)
    vif_data = pd.DataFrame({'feature': feature_cols, 'VIF': np.nan})
    
    if calculate_vif:
        print("\nCalculating VIF (multicollinearity check)...")
        try:
            vif_values = [variance_inflation_factor(X.values, i) for i in range(len(feature_cols))]
            vif_data = pd.DataFrame({
                'feature': feature_cols,
                'VIF': vif_values
            })
            vif_data = vif_data.sort_values('VIF', ascending=False)
            
            high_vif = vif_data[vif_data['VIF'] > 10]
            if len(high_vif) > 0:
                print(f"⚠️  {len(high_vif)} features with VIF > 10 (multicollinearity)")
            else:
                print("✓ No severe multicollinearity (all VIF < 10)")
        except Exception as e:
            print(f"⚠️  VIF calculation failed: {e}")
    
    # Fit OLS model
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    print(f"\nModel Performance:")
    print(f"  R²: {model.rsquared:.4f}")
    print(f"  Adj R²: {model.rsquared_adj:.4f}")
    print(f"  F-statistic p-value: {model.f_pvalue:.4e}")
    
    # Extract coefficients
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.params[1:].values,
        'std_error': model.bse[1:].values,
        'p_value': model.pvalues[1:].values,
        't_stat': model.tvalues[1:].values,
    })
    
    # Calculate importance (absolute coefficient)
    importance_df['importance'] = importance_df['coefficient'].abs()
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Count significant features
    sig_count = (importance_df['p_value'] < 0.05).sum()
    print(f"  Significant features (p < 0.05): {sig_count}/{len(feature_cols)}")
    
    print("\nTop 10 Features (by |coefficient|):")
    print(importance_df.head(10)[['feature', 'coefficient', 'p_value']].to_string(index=False))
    
    return model, importance_df, vif_data


# =============================================================================
# RANDOM FOREST
# =============================================================================

def build_random_forest_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str,
    user_id_var: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    n_splits: int = 5,
) -> Tuple[Any, pd.DataFrame, Dict[str, List[float]]]:
    """
    Build Random Forest with cluster-aware cross-validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and outcome
    feature_cols : List[str]
        List of feature column names
    outcome_var : str
        Target variable name
    user_id_var : str
        User identifier for GroupKFold
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum tree depth
    n_splits : int
        Number of CV folds
    
    Returns:
    --------
    Tuple of (model, importance_df, cv_scores)
        - model: fitted RandomForest model (best fold)
        - importance_df: DataFrame with feature importances
        - cv_scores: Dict with CV metrics per fold
    """
    print("\n" + "="*70)
    print("RANDOM FOREST")
    print("="*70)

    X = df[feature_cols].copy()
    y = df[outcome_var].copy()
    groups = df[user_id_var].copy()

    # Handle missing values
    mask = X.notna().all(axis=1) & y.notna() & groups.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    groups = groups[mask].reset_index(drop=True)

    print(f"\nObservations: {len(y)}")
    print(f"Unique users: {groups.nunique()}")
    print(f"Features: {len(feature_cols)}")

    # Cross-validation
    print(f"\nRunning {n_splits}-fold cluster-based CV...")
    gkf = GroupKFold(n_splits=n_splits)
    
    cv_scores = {
        'train_r2': [], 'test_r2': [], 
        'test_mse': [], 'test_mae': []
    }
    fold_models = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf_fold = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=1
        )
        rf_fold.fit(X_train, y_train)
        fold_models.append(rf_fold)

        # Calculate metrics
        train_r2 = r2_score(y_train, rf_fold.predict(X_train))
        test_r2 = r2_score(y_test, rf_fold.predict(X_test))
        test_mse = mean_squared_error(y_test, rf_fold.predict(X_test))
        test_mae = mean_absolute_error(y_test, rf_fold.predict(X_test))

        cv_scores['train_r2'].append(train_r2)
        cv_scores['test_r2'].append(test_r2)
        cv_scores['test_mse'].append(test_mse)
        cv_scores['test_mae'].append(test_mae)

        print(f"  Fold {fold}: Test R²={test_r2:.4f}, Train R²={train_r2:.4f}")

    # Summary
    print(f"\nCV Summary:")
    print(f"  Test R²: {np.mean(cv_scores['test_r2']):.4f} ± {np.std(cv_scores['test_r2']):.4f}")
    print(f"  Test RMSE: {np.sqrt(np.mean(cv_scores['test_mse'])):.4f}")
    print(f"  Test MAE: {np.mean(cv_scores['test_mae']):.4f}")
    
    # Overfitting check
    overfit_gap = np.mean(cv_scores['train_r2']) - np.mean(cv_scores['test_r2'])
    if overfit_gap > 0.1:
        print(f"  ⚠️ Possible overfitting (gap = {overfit_gap:.4f})")
    else:
        print(f"  ✓ Overfitting check OK (gap = {overfit_gap:.4f})")

    # Use best fold model
    best_idx = np.argmax(cv_scores['test_r2'])
    rf_model = fold_models[best_idx]
    print(f"\nUsing best fold model (Fold {best_idx + 1})")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf_model.feature_importances_,
    })
    importance_df['rf_importance_pct'] = (importance_df['rf_importance'] * 100).round(2)
    importance_df = importance_df.sort_values('rf_importance', ascending=False)

    print("\nTop 10 Features (by RF importance):")
    print(importance_df.head(10)[['feature', 'rf_importance_pct']].to_string(index=False))

    return rf_model, importance_df, cv_scores


# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

def build_logistic_regression_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str,
    user_id_var: str,
    significance_level: float = 0.05,
    regularization_C: float = 1.0,
    n_splits: int = 5,
) -> Tuple[Any, pd.DataFrame, Dict[str, List[float]]]:
    """
    Build Logistic Regression with L2 regularization and cluster-aware CV.
    
    Uses sklearn LogisticRegression with regularization to prevent
    coefficient explosion (infinite coefficients) from separation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and outcome
    feature_cols : List[str]
        List of feature column names
    outcome_var : str
        Target variable (will be binarized: >0 = 1)
    user_id_var : str
        User identifier for GroupKFold
    significance_level : float
        Alpha for significance testing
    regularization_C : float
        Inverse regularization strength (smaller = stronger regularization)
    n_splits : int
        Number of CV folds
    
    Returns:
    --------
    Tuple of (model, importance_df, cv_scores)
        - model: fitted LogisticRegression model
        - importance_df: DataFrame with coefficients, odds ratios, p-values
        - cv_scores: Dict with CV metrics per fold
    """
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION (L2 Regularized)")
    print("="*70)

    X = df[feature_cols].copy()
    y_raw = df[outcome_var].copy()
    groups = df[user_id_var].copy()

    # Binarize outcome
    y = (y_raw.fillna(0) > 0).astype(int)

    # Handle missing values
    mask = X.notna().all(axis=1) & y.notna() & groups.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    groups = groups[mask].reset_index(drop=True)

    print(f"\nObservations: {len(y)}")
    print(f"Unique users: {groups.nunique()}")
    print(f"Positive class ({outcome_var} > 0): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Negative class: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    # Fit model
    print(f"\nFitting Logistic Regression (C={regularization_C})...")
    lr_model = LogisticRegression(
        penalty='l2',
        C=regularization_C,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
    )
    lr_model.fit(X_scaled, y)

    # Cross-validation
    print(f"\nRunning {n_splits}-fold cluster-based CV...")
    gkf = GroupKFold(n_splits=n_splits)
    
    cv_scores = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'auc': [], 'train_accuracy': []
    }

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups), 1):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = LogisticRegression(
            penalty='l2', C=regularization_C, solver='lbfgs',
            max_iter=1000, random_state=42
        )
        fold_model.fit(X_train, y_train)

        y_train_pred = fold_model.predict(X_train)
        y_test_pred = fold_model.predict(X_test)
        y_test_prob = fold_model.predict_proba(X_test)[:, 1]

        cv_scores['train_accuracy'].append(accuracy_score(y_train, y_train_pred))
        cv_scores['accuracy'].append(accuracy_score(y_test, y_test_pred))
        cv_scores['precision'].append(precision_score(y_test, y_test_pred, zero_division=0))
        cv_scores['recall'].append(recall_score(y_test, y_test_pred, zero_division=0))
        cv_scores['f1'].append(f1_score(y_test, y_test_pred, zero_division=0))
        
        try:
            cv_scores['auc'].append(roc_auc_score(y_test, y_test_prob))
        except:
            cv_scores['auc'].append(np.nan)

        print(f"  Fold {fold}: Acc={cv_scores['accuracy'][-1]:.4f}, AUC={cv_scores['auc'][-1]:.4f}")

    # Summary
    print(f"\nCV Summary:")
    print(f"  Accuracy:  {np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}")
    print(f"  Precision: {np.mean(cv_scores['precision']):.4f} ± {np.std(cv_scores['precision']):.4f}")
    print(f"  Recall:    {np.mean(cv_scores['recall']):.4f} ± {np.std(cv_scores['recall']):.4f}")
    print(f"  F1:        {np.mean(cv_scores['f1']):.4f} ± {np.std(cv_scores['f1']):.4f}")
    print(f"  AUC:       {np.nanmean(cv_scores['auc']):.4f} ± {np.nanstd(cv_scores['auc']):.4f}")

    # Extract coefficients (adjusted for scaling)
    raw_coefs = lr_model.coef_[0]
    adjusted_coefs = raw_coefs / scaler.scale_  # Per-unit-of-original-feature
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'lr_coefficient': adjusted_coefs,
        'lr_odds_ratio': np.exp(adjusted_coefs),
    })
    
    # Approximate p-values using Wald test
    try:
        n = len(y)
        y_prob = lr_model.predict_proba(X_scaled)[:, 1]
        W = np.diag(y_prob * (1 - y_prob))
        X_np = X_scaled.values
        cov_matrix = np.linalg.inv(X_np.T @ W @ X_np)
        std_errors = np.sqrt(np.diag(cov_matrix)) / scaler.scale_
        z_scores = adjusted_coefs / std_errors
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    except:
        std_errors = np.full(len(feature_cols), np.nan)
        z_scores = np.full(len(feature_cols), np.nan)
        p_values = np.full(len(feature_cols), np.nan)
    
    importance_df['lr_std_error'] = std_errors
    importance_df['lr_z_stat'] = z_scores
    importance_df['lr_p_value'] = p_values
    importance_df['lr_significant'] = importance_df['lr_p_value'] < significance_level
    importance_df['lr_importance'] = importance_df['lr_coefficient'].abs()
    importance_df = importance_df.sort_values('lr_importance', ascending=False)

    print("\nTop 10 Features (by |coefficient|):")
    print(importance_df.head(10)[['feature', 'lr_coefficient', 'lr_odds_ratio']].to_string(index=False))

    # Store scaler for later use
    lr_model.scaler_ = scaler
    
    return lr_model, importance_df, cv_scores


# =============================================================================
# HOLDOUT TEST SET & ENSEMBLE FUNCTIONS
# =============================================================================

def create_user_level_train_test_split(
    df: pd.DataFrame,
    user_id_var: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by user to prevent data leakage.

    Ensures no user appears in both train and test sets.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with user identifier
    user_id_var : str
        Column name for user identifier
    test_size : float
        Proportion of users to include in test set
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    Tuple of (train_df, test_df)
    """
    np.random.seed(random_state)

    unique_users = df[user_id_var].unique()
    n_test_users = int(len(unique_users) * test_size)

    # Randomly select test users
    test_users = np.random.choice(unique_users, size=n_test_users, replace=False)

    train_df = df[~df[user_id_var].isin(test_users)].reset_index(drop=True)
    test_df = df[df[user_id_var].isin(test_users)].reset_index(drop=True)

    print(f"Train/Test Split (by user):")
    print(f"  Train: {len(train_df)} obs, {train_df[user_id_var].nunique()} users")
    print(f"  Test:  {len(test_df)} obs, {test_df[user_id_var].nunique()} users")

    return train_df, test_df


def evaluate_on_holdout(
    model: Any,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str,
    model_type: str = 'regression',
    scaler: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained model on holdout test set.

    Parameters:
    -----------
    model : Any
        Trained model with predict method
    test_df : pd.DataFrame
        Holdout test data
    feature_cols : List[str]
        Feature columns
    outcome_var : str
        Target variable
    model_type : str
        'regression' or 'classification'
    scaler : Any, optional
        Fitted scaler for logistic regression

    Returns:
    --------
    Dict with performance metrics
    """
    X_test = test_df[feature_cols].copy()
    y_test = test_df[outcome_var].copy()

    # Handle missing values
    mask = X_test.notna().all(axis=1) & y_test.notna()
    X_test = X_test[mask].reset_index(drop=True)
    y_test = y_test[mask].reset_index(drop=True)

    # Scale if needed (for logistic regression)
    if scaler is not None:
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    if model_type == 'regression':
        y_pred = model.predict(X_test)

        metrics = {
            'holdout_r2': r2_score(y_test, y_pred),
            'holdout_mse': mean_squared_error(y_test, y_pred),
            'holdout_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'holdout_mae': mean_absolute_error(y_test, y_pred),
            'n_test': len(y_test),
        }

        print(f"Holdout Performance (Regression):")
        print(f"  R²:   {metrics['holdout_r2']:.4f}")
        print(f"  RMSE: {metrics['holdout_rmse']:.4f}")
        print(f"  MAE:  {metrics['holdout_mae']:.4f}")

    else:  # classification
        y_test_binary = (y_test > 0).astype(int)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'holdout_accuracy': accuracy_score(y_test_binary, y_pred),
            'holdout_precision': precision_score(y_test_binary, y_pred, zero_division=0),
            'holdout_recall': recall_score(y_test_binary, y_pred, zero_division=0),
            'holdout_f1': f1_score(y_test_binary, y_pred, zero_division=0),
            'n_test': len(y_test),
        }

        if y_prob is not None:
            try:
                metrics['holdout_auc'] = roc_auc_score(y_test_binary, y_prob)
            except:
                metrics['holdout_auc'] = np.nan

        print(f"Holdout Performance (Classification):")
        print(f"  Accuracy:  {metrics['holdout_accuracy']:.4f}")
        print(f"  Precision: {metrics['holdout_precision']:.4f}")
        print(f"  Recall:    {metrics['holdout_recall']:.4f}")
        print(f"  F1:        {metrics['holdout_f1']:.4f}")
        if 'holdout_auc' in metrics:
            print(f"  AUC:       {metrics['holdout_auc']:.4f}")

    return metrics


class EnsembleRandomForest:
    """
    Ensemble of Random Forest models from cross-validation folds.

    Averages predictions across all fold models for more stable,
    robust predictions than using a single best-fold model.

    Attributes:
    -----------
    models : List
        List of fitted RF models from each fold
    feature_cols : List[str]
        Feature column names
    feature_importances_ : np.ndarray
        Averaged feature importances across all models
    """

    def __init__(self, models: List, feature_cols: List[str]):
        """Initialize with list of trained models."""
        self.models = models
        self.feature_cols = feature_cols

        # Average feature importances
        importances = np.array([m.feature_importances_ for m in models])
        self.feature_importances_ = importances.mean(axis=0)
        self.feature_importances_std_ = importances.std(axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict by averaging predictions from all fold models.

        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on

        Returns:
        --------
        np.ndarray of averaged predictions
        """
        predictions = np.array([m.predict(X) for m in self.models])
        return predictions.mean(axis=0)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty estimates from model disagreement.

        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on

        Returns:
        --------
        Dict with 'mean', 'std', 'lower', 'upper' predictions
        """
        predictions = np.array([m.predict(X) for m in self.models])
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower': mean_pred - 1.96 * std_pred,  # 95% CI
            'upper': mean_pred + 1.96 * std_pred,
        }

    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get feature importances with uncertainty."""
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.feature_importances_,
            'importance_std': self.feature_importances_std_,
            'importance_pct': (self.feature_importances_ * 100).round(2),
        }).sort_values('importance', ascending=False)


def build_ensemble_random_forest(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str,
    user_id_var: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    n_splits: int = 5,
) -> Tuple[EnsembleRandomForest, pd.DataFrame, Dict[str, List[float]]]:
    """
    Build an ensemble Random Forest using all CV fold models.

    Unlike build_random_forest_model which returns the best fold model,
    this returns an EnsembleRandomForest that averages predictions
    from all fold models for more stable predictions.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and outcome
    feature_cols : List[str]
        List of feature column names
    outcome_var : str
        Target variable name
    user_id_var : str
        User identifier for GroupKFold
    n_estimators : int
        Number of trees per model
    max_depth : int
        Maximum tree depth
    n_splits : int
        Number of CV folds

    Returns:
    --------
    Tuple of (ensemble_model, importance_df, cv_scores)
    """
    print("\n" + "="*70)
    print("ENSEMBLE RANDOM FOREST")
    print("="*70)

    X = df[feature_cols].copy()
    y = df[outcome_var].copy()
    groups = df[user_id_var].copy()

    # Handle missing values
    mask = X.notna().all(axis=1) & y.notna() & groups.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    groups = groups[mask].reset_index(drop=True)

    print(f"\nObservations: {len(y)}")
    print(f"Unique users: {groups.nunique()}")
    print(f"Features: {len(feature_cols)}")

    # Cross-validation
    print(f"\nTraining {n_splits} models (one per fold)...")
    gkf = GroupKFold(n_splits=n_splits)

    cv_scores = {
        'train_r2': [], 'test_r2': [],
        'test_mse': [], 'test_mae': []
    }
    fold_models = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf_fold = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42 + fold,  # Different seed per fold for diversity
            n_jobs=-1
        )
        rf_fold.fit(X_train, y_train)
        fold_models.append(rf_fold)

        # Calculate metrics
        train_r2 = r2_score(y_train, rf_fold.predict(X_train))
        test_r2 = r2_score(y_test, rf_fold.predict(X_test))
        test_mse = mean_squared_error(y_test, rf_fold.predict(X_test))
        test_mae = mean_absolute_error(y_test, rf_fold.predict(X_test))

        cv_scores['train_r2'].append(train_r2)
        cv_scores['test_r2'].append(test_r2)
        cv_scores['test_mse'].append(test_mse)
        cv_scores['test_mae'].append(test_mae)

        print(f"  Fold {fold}: Test R²={test_r2:.4f}, Train R²={train_r2:.4f}")

    # Create ensemble
    ensemble = EnsembleRandomForest(fold_models, feature_cols)

    # Summary
    print(f"\nCV Summary:")
    print(f"  Test R²: {np.mean(cv_scores['test_r2']):.4f} ± {np.std(cv_scores['test_r2']):.4f}")
    print(f"  Test RMSE: {np.sqrt(np.mean(cv_scores['test_mse'])):.4f}")
    print(f"  Test MAE: {np.mean(cv_scores['test_mae']):.4f}")
    print(f"\n✓ Ensemble created with {len(fold_models)} models")

    # Overfitting check
    overfit_gap = np.mean(cv_scores['train_r2']) - np.mean(cv_scores['test_r2'])
    if overfit_gap > 0.1:
        print(f"  ⚠️ Possible overfitting (gap = {overfit_gap:.4f})")
    else:
        print(f"  ✓ Overfitting check OK (gap = {overfit_gap:.4f})")

    # Get feature importance from ensemble
    importance_df = ensemble.get_feature_importance_df()
    importance_df = importance_df.rename(columns={
        'importance': 'rf_importance',
        'importance_std': 'rf_importance_std',
        'importance_pct': 'rf_importance_pct',
    })

    print("\nTop 10 Features (by ensemble importance):")
    print(importance_df.head(10)[['feature', 'rf_importance_pct', 'rf_importance_std']].to_string(index=False))

    return ensemble, importance_df, cv_scores


# =============================================================================
# RUN ALL MODELS
# =============================================================================

def run_all_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_var: str = "wonky_study_count",
    user_id_var: str = "respondentPk",
    include_logistic: bool = True,
    include_vif: bool = True,
    rf_n_estimators: int = 100,
    lr_regularization_C: float = 1.0,
) -> Dict:
    """
    Run all models and create unified comparison table.
    
    This is the main entry point for the modelling pipeline.
    Runs Linear, Random Forest, and optionally Logistic regression,
    then creates a comparison table with rankings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and outcome
    feature_cols : List[str]
        List of feature column names
    outcome_var : str
        Target variable name
    user_id_var : str
        User identifier for clustering
    include_logistic : bool
        Whether to include logistic regression
    include_vif : bool
        Whether to calculate VIF
    rf_n_estimators : int
        Number of trees for Random Forest
    lr_regularization_C : float
        Regularization strength for Logistic Regression
    
    Returns:
    --------
    Dict with keys:
        - linear_model, linear_importance, vif_data
        - rf_model, rf_importance, rf_cv_results
        - lr_model, lr_importance, lr_cv_results (if include_logistic)
        - comparison: merged comparison table
        - feature_cols: cleaned feature list
    """
    from modelling.modelling_utils import remove_collinear_features
    
    print("\n" + "="*80)
    print("RUNNING ALL MODELS")
    print("="*80)

    # Clean features
    feature_cols_clean = remove_collinear_features(feature_cols)

    print(f"\nDataset: {len(df)} observations")
    print(f"Features: {len(feature_cols_clean)}")
    print(f"Outcome: {outcome_var}")
    print(f"Unique users: {df[user_id_var].nunique()}")

    # 1. Linear Regression
    linear_model, linear_importance, vif_data = build_linear_baseline(
        df, feature_cols_clean, outcome_var, user_id_var, calculate_vif=include_vif
    )

    # 2. Random Forest
    rf_model, rf_importance, rf_cv_results = build_random_forest_model(
        df, feature_cols_clean, outcome_var, user_id_var, n_estimators=rf_n_estimators
    )

    # 3. Logistic Regression (optional)
    lr_model, lr_importance, lr_cv_results = None, None, None
    if include_logistic:
        lr_model, lr_importance, lr_cv_results = build_logistic_regression_model(
            df, feature_cols_clean, outcome_var, user_id_var,
            regularization_C=lr_regularization_C
        )

    # =========================================================================
    # CREATE COMPARISON TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("CREATING COMPARISON TABLE")
    print("="*70)

    # Start with linear results
    comparison = linear_importance[['feature', 'coefficient', 'p_value', 't_stat']].copy()
    comparison = comparison.rename(columns={
        'coefficient': 'linear_coef',
        'p_value': 'linear_p_value',
        't_stat': 'linear_t_stat',
    })

    # Add RF importance
    comparison = comparison.merge(
        rf_importance[['feature', 'rf_importance', 'rf_importance_pct']],
        on='feature', how='left'
    )

    # Add logistic results
    if include_logistic and lr_importance is not None:
        comparison = comparison.merge(
            lr_importance[['feature', 'lr_coefficient', 'lr_odds_ratio', 
                          'lr_p_value', 'lr_significant']],
            on='feature', how='left'
        )

    # Add VIF
    comparison = comparison.merge(vif_data[['feature', 'VIF']], on='feature', how='left')

    # Calculate ranks
    comparison['linear_rank'] = comparison['linear_coef'].abs().rank(ascending=False)
    comparison['rf_rank'] = comparison['rf_importance'].rank(ascending=False)
    
    if include_logistic and 'lr_coefficient' in comparison.columns:
        comparison['lr_rank'] = comparison['lr_coefficient'].abs().rank(ascending=False)
        comparison['avg_rank'] = (
            comparison['linear_rank'] + comparison['rf_rank'] + comparison['lr_rank']
        ) / 3
    else:
        comparison['avg_rank'] = (comparison['linear_rank'] + comparison['rf_rank']) / 2

    comparison = comparison.sort_values('avg_rank')

    print(f"\n✓ Comparison table created with {len(comparison)} features")

    # Build results dictionary
    results = {
        'linear_model': linear_model,
        'linear_importance': linear_importance,
        'vif_data': vif_data,
        'rf_model': rf_model,
        'rf_importance': rf_importance,
        'rf_cv_results': rf_cv_results,
        'comparison': comparison,
        'feature_cols': feature_cols_clean,
    }

    if include_logistic:
        results['lr_model'] = lr_model
        results['lr_importance'] = lr_importance
        results['lr_cv_results'] = lr_cv_results

    return results