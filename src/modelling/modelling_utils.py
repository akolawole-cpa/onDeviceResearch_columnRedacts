"""
Modelling Utilities Module

Helper functions for modelling including:
- Feature preprocessing (collinearity removal)
- SHAP analysis
- Stakeholder reporting
- Visualization helpers

Usage:
    from modelling.modelling_utils import (
        remove_collinear_features,
        run_shap_analysis,
        create_stakeholder_report,
        print_model_summary,
    )
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Tuple, Any, Optional

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE PREPROCESSING
# =============================================================================

def remove_collinear_features(
    feature_list: List[str],
    additional_removals: Optional[List[str]] = None,
) -> List[str]:
    """
    Remove redundant features that cause perfect multicollinearity.
    
    Parameters:
    -----------
    feature_list : List[str]
        Original feature list
    additional_removals : List[str], optional
        Additional features to remove
    
    Returns:
    --------
    List[str]
        Cleaned feature list
    """
    # Default features to remove (known collinear)
    features_to_remove = [
        'is_weekend',        # Redundant: = is_saturday + is_sunday
        'is_business_hour',  # Redundant: combination of weekday/weekend versions
    ]
    
    # Add any additional removals
    if additional_removals:
        features_to_remove.extend(additional_removals)
    
    # Remove duplicates
    features_to_remove = list(set(features_to_remove))
    
    # Filter
    cleaned_features = [f for f in feature_list if f not in features_to_remove]
    
    removed = [f for f in features_to_remove if f in feature_list]
    if removed:
        print(f"Removed {len(removed)} collinear features: {removed}")
    print(f"Features: {len(feature_list)} → {len(cleaned_features)}")
    
    return cleaned_features


def filter_rare_features(
    X: pd.DataFrame,
    min_prevalence: float = 0.01,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Filter out features that are too rare (cause separation in logistic regression).
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    min_prevalence : float
        Minimum proportion of 1s required (default 1%)
    
    Returns:
    --------
    Tuple of (valid_features, rare_features)
        - valid_features: List of features that pass filter
        - rare_features: List of (feature, prevalence) tuples that failed
    """
    valid_features = []
    rare_features = []
    
    for col in X.columns:
        prevalence = X[col].mean()
        if prevalence >= min_prevalence and prevalence <= (1 - min_prevalence):
            valid_features.append(col)
        else:
            rare_features.append((col, prevalence))
    
    if rare_features:
        print(f"\n⚠️  {len(rare_features)} rare features filtered:")
        for feat, prev in rare_features[:5]:
            print(f"    {feat}: {prev*100:.2f}%")
        if len(rare_features) > 5:
            print(f"    ... and {len(rare_features) - 5} more")
    
    return valid_features, rare_features


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def run_shap_analysis(
    results: Dict,
    df: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Dict:
    """
    Run SHAP analysis on Random Forest model.
    
    SHAP provides:
    - Direction of feature effects (not just magnitude)
    - Per-observation explanations
    - Better feature importance than MDI
    
    Parameters:
    -----------
    results : Dict
        Output from run_all_models()
    df : pd.DataFrame
        DataFrame with features
    sample_size : int
        Number of samples for SHAP (for speed)
    random_state : int
        For reproducibility
    
    Returns:
    --------
    Dict with SHAP results added
    """
    print("\n" + "="*70)
    print("🔮 SHAP ANALYSIS")
    print("="*70)
    
    rf_model = results['rf_model']
    feature_cols = results['feature_cols']
    
    # Prepare data
    X = df[feature_cols].copy()
    mask = X.notna().all(axis=1)
    X = X[mask].reset_index(drop=True)
    
    # Sample if large
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
        print(f"Using {sample_size} samples (full: {len(X)})")
    else:
        X_sample = X
        print(f"Using full dataset ({len(X)} samples)")
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    print("✓ Done!")
    
    # Create importance DataFrame
    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': np.abs(shap_values).mean(axis=0),
        'shap_mean': shap_values.mean(axis=0),
    })
    
    # Add direction interpretation
    shap_importance['shap_direction'] = shap_importance['shap_mean'].apply(
        lambda x: '↑ Increases' if x > 0.001 else ('↓ Decreases' if x < -0.001 else '→ Neutral')
    )
    
    shap_importance['shap_rank'] = shap_importance['shap_importance'].rank(ascending=False)
    shap_importance = shap_importance.sort_values('shap_importance', ascending=False)
    
    # Store in results
    results['shap_values'] = shap_values
    results['shap_explainer'] = explainer
    results['shap_importance'] = shap_importance
    results['shap_X_sample'] = X_sample
    
    # Update comparison table
    if 'comparison' in results:
        results['comparison'] = results['comparison'].merge(
            shap_importance[['feature', 'shap_importance', 'shap_mean', 
                            'shap_direction', 'shap_rank']],
            on='feature', how='left'
        )
        
        # Recalculate avg_rank with SHAP
        if 'lr_rank' in results['comparison'].columns:
            results['comparison']['avg_rank'] = (
                results['comparison']['linear_rank'] +
                results['comparison']['rf_rank'] +
                results['comparison']['lr_rank'] +
                results['comparison']['shap_rank']
            ) / 4
        else:
            results['comparison']['avg_rank'] = (
                results['comparison']['linear_rank'] +
                results['comparison']['rf_rank'] +
                results['comparison']['shap_rank']
            ) / 3
        
        results['comparison'] = results['comparison'].sort_values('avg_rank')
    
    print(f"\n✓ SHAP analysis complete")
    print("\nTop 10 Features (by SHAP importance):")
    print(shap_importance.head(10)[['feature', 'shap_importance', 'shap_direction']].to_string(index=False))
    
    return results


def plot_shap_summary(results: Dict, max_display: int = 20):
    """
    Create SHAP summary plot (beeswarm).
    
    Shows feature importance and direction of effect.
    """
    if 'shap_values' not in results:
        print("⚠️ Run run_shap_analysis() first")
        return
    
    print("\n📊 SHAP Summary Plot")
    print("  • Y-axis: Features (top = most important)")
    print("  • X-axis: Impact on prediction")
    print("  • Color: Red = high feature value, Blue = low")
    
    shap.summary_plot(
        results['shap_values'], 
        results['shap_X_sample'], 
        max_display=max_display, 
        show=True
    )


def plot_shap_bar(results: Dict, max_display: int = 20):
    """Create SHAP bar plot (simpler view)."""
    if 'shap_values' not in results:
        print("⚠️ Run run_shap_analysis() first")
        return
    
    shap.summary_plot(
        results['shap_values'], 
        results['shap_X_sample'], 
        plot_type="bar", 
        max_display=max_display, 
        show=True
    )


def plot_shap_dependence(
    results: Dict, 
    feature: str, 
    interaction_feature: Optional[str] = None
):
    """
    Create SHAP dependence plot for a specific feature.
    
    Shows how feature value affects predictions.
    """
    if 'shap_values' not in results:
        print("⚠️ Run run_shap_analysis() first")
        return
    
    X_sample = results['shap_X_sample']
    
    if feature not in X_sample.columns:
        print(f"⚠️ Feature '{feature}' not found")
        return
    
    print(f"\n📊 SHAP Dependence: {feature}")
    
    if interaction_feature:
        shap.dependence_plot(
            feature, results['shap_values'], X_sample,
            interaction_index=interaction_feature, show=True
        )
    else:
        shap.dependence_plot(
            feature, results['shap_values'], X_sample, show=True
        )


# =============================================================================
# STAKEHOLDER REPORTING
# =============================================================================

def interpret_odds_ratio(or_value: float) -> str:
    """Convert odds ratio to plain English."""
    if pd.isna(or_value) or or_value <= 0:
        return "Invalid"
    
    if or_value > 100 or or_value < 0.01:
        return "⚠️ Extreme"
    
    if abs(or_value - 1) < 0.05:
        return "No effect"
    
    if or_value > 1:
        pct = (or_value - 1) * 100
        if pct > 100:
            return f"↑ {or_value:.1f}x more likely"
        return f"↑ {pct:.0f}% more likely"
    else:
        pct = (1 - or_value) * 100
        return f"↓ {pct:.0f}% less likely"


def create_stakeholder_report(results: Dict) -> pd.DataFrame:
    """
    Create a clean, presentation-ready summary table.
    
    Parameters:
    -----------
    results : Dict
        Output from run_all_models()
    
    Returns:
    --------
    pd.DataFrame
        Formatted table for stakeholders
    """
    comp = results['comparison'].copy()
    
    report = pd.DataFrame()
    report['Feature'] = comp['feature']
    
    # Linear: coefficient with significance stars
    def format_linear(row):
        coef = row['linear_coef']
        p = row['linear_p_value']
        stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        return f"{coef:+.4f}{stars}"
    
    report['Linear Effect'] = comp.apply(format_linear, axis=1)
    report['Linear p-value'] = comp['linear_p_value'].apply(
        lambda x: f"{x:.4f}" if x >= 0.0001 else f"{x:.2e}"
    )
    
    # RF importance
    report['RF Importance %'] = comp['rf_importance_pct'].apply(lambda x: f"{x:.2f}%")
    
    # Logistic (if available)
    if 'lr_odds_ratio' in comp.columns:
        report['Odds Ratio'] = comp['lr_odds_ratio'].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) and 0.01 <= x <= 100 else "⚠️"
        )
        report['OR Interpretation'] = comp['lr_odds_ratio'].apply(interpret_odds_ratio)
    
    # SHAP (if available)
    if 'shap_direction' in comp.columns:
        report['SHAP Direction'] = comp['shap_direction']
    
    # Ranks and VIF
    report['Consensus Rank'] = comp['avg_rank'].round(1)
    report['VIF'] = comp['VIF'].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
    )
    report['VIF Flag'] = comp['VIF'].apply(
        lambda x: '⚠️' if pd.notna(x) and x > 10 else ('⚡' if pd.notna(x) and x > 5 else '✓')
    )
    
    return report.sort_values('Consensus Rank')


def print_model_summary(results: Dict):
    """
    Print comprehensive model summary for stakeholders.
    """
    print("="*80)
    print("📊 MODEL RESULTS SUMMARY")
    print("="*80)
    
    # Linear Model
    lm = results['linear_model']
    print(f"\n1. LINEAR REGRESSION")
    print(f"   R²: {lm.rsquared:.4f}")
    print(f"   Significant features: {(results['linear_importance']['p_value'] < 0.05).sum()}")
    
    # Random Forest
    rf_cv = results['rf_cv_results']
    print(f"\n2. RANDOM FOREST")
    print(f"   Test R²: {np.mean(rf_cv['test_r2']):.4f} ± {np.std(rf_cv['test_r2']):.4f}")
    print(f"   Test RMSE: {np.sqrt(np.mean(rf_cv['test_mse'])):.4f}")
    
    # Logistic Regression
    if 'lr_cv_results' in results:
        lr_cv = results['lr_cv_results']
        print(f"\n3. LOGISTIC REGRESSION")
        print(f"   Accuracy: {np.mean(lr_cv['accuracy']):.4f} ± {np.std(lr_cv['accuracy']):.4f}")
        print(f"   AUC-ROC:  {np.nanmean(lr_cv['auc']):.4f} ± {np.nanstd(lr_cv['auc']):.4f}")
    
    # SHAP
    if 'shap_importance' in results:
        print(f"\n4. SHAP ANALYSIS")
        print(f"   ✓ SHAP values calculated")
    
    # Top features
    comp = results['comparison']
    print(f"\n🏆 TOP 5 CONSENSUS FEATURES:")
    for i, (_, row) in enumerate(comp.head(5).iterrows(), 1):
        print(f"   {i}. {row['feature']} (rank: {row['avg_rank']:.1f})")


def get_significant_features(
    results: Dict,
    p_threshold: float = 0.05,
    min_rf_importance: float = 0.01,
) -> List[str]:
    """
    Get features that are significant across models.
    
    Parameters:
    -----------
    results : Dict
        Output from run_all_models()
    p_threshold : float
        P-value threshold for significance
    min_rf_importance : float
        Minimum RF importance (as decimal)
    
    Returns:
    --------
    List[str]
        Features meeting criteria
    """
    comp = results['comparison']
    
    # Start with linear significance
    sig_linear = set(comp[comp['linear_p_value'] < p_threshold]['feature'])
    
    # RF importance
    top_rf = set(comp[comp['rf_importance'] >= min_rf_importance]['feature'])
    
    # Intersection
    if 'lr_significant' in comp.columns:
        sig_lr = set(comp[comp['lr_significant'] == True]['feature'])
        high_confidence = sig_linear & top_rf & sig_lr
    else:
        high_confidence = sig_linear & top_rf
    
    return list(high_confidence)


def export_results(
    results: Dict,
    output_path: str = 'feature_importance_results.csv',
) -> pd.DataFrame:
    """
    Export results to CSV for further analysis.
    
    Returns the export DataFrame.
    """
    comp = results['comparison'].copy()
    
    export_cols = ['feature', 'linear_coef', 'linear_p_value', 
                   'rf_importance', 'rf_importance_pct', 'avg_rank', 'VIF']
    
    if 'lr_coefficient' in comp.columns:
        export_cols.extend(['lr_coefficient', 'lr_odds_ratio', 'lr_p_value'])
    
    if 'shap_importance' in comp.columns:
        export_cols.extend(['shap_importance', 'shap_mean', 'shap_direction'])
    
    export_df = comp[export_cols].copy()
    export_df.to_csv(output_path, index=False)
    
    print(f"✓ Results exported to {output_path}")
    
    return export_df