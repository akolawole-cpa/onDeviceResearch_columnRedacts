"""
Modelling Utilities Module

Helper functions for modelling including:
- Feature preprocessing (collinearity removal, VIF analysis)
- SHAP analysis and feature interactions
- Report generation and executive summaries
- Model comparison scorecard
- Calibration analysis

Usage:
    from modelling.modelling_utils import (
        remove_collinear_features,
        calculate_vif_detailed,
        run_shap_analysis,
        analyze_feature_interactions,
        create_report,
        create_model_scorecard,
        analyze_logistic_calibration,
        generate_modelling_executive_summary,
        print_model_summary,
        get_significant_features,
        export_results,
        plot_shap_summary,
        plot_shap_bar,
        plot_shap_dependence,
    )
"""

import pandas as pd
import numpy as np
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Tuple, Any, Optional

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Outside model VIF analysis
# =============================================================================


def calculate_vif_detailed(
    df: pd.DataFrame,
    feature_cols: List[str],
    vif_threshold_high: float = 10.0,
    vif_threshold_moderate: float = 5.0,
) -> pd.DataFrame:
    """
    Calculate VIF for all features with detailed diagnostics.
    
    VIF (Variance Inflation Factor) measures multicollinearity:
    - VIF = 1: No correlation with other features
    - VIF < 5: Low correlation (acceptable)
    - VIF 5-10: Moderate correlation (monitor)
    - VIF > 10: High correlation (consider removal)
    - VIF = inf: Perfect multicollinearity (must remove)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    feature_cols : List[str]
        Feature columns to analyze
    vif_threshold_high : float
        Threshold for high VIF warning (default 10)
    vif_threshold_moderate : float
        Threshold for moderate VIF warning (default 5)
    
    Returns:
    --------
    pd.DataFrame
        VIF results with flags and recommendations
    """
    X = df[feature_cols].copy()
    X_clean = X.dropna()
    
    print(f"Calculating VIF for {len(feature_cols)} features...")
    print(f"Using {len(X_clean)} complete observations")
    
    vif_results = []
    for i, col in enumerate(feature_cols):
        try:
            vif_value = variance_inflation_factor(X_clean.values, i)
        except Exception as e:
            vif_value = np.inf
        vif_results.append({'feature': col, 'VIF': vif_value})
    
    vif_df = pd.DataFrame(vif_results)
    
    # Add diagnostic columns
    vif_df['vif_flag'] = vif_df['VIF'].apply(
        lambda x: '🚨 INFINITE' if np.isinf(x) 
        else ('⚠️ HIGH' if x > vif_threshold_high 
        else ('⚡ MODERATE' if x > vif_threshold_moderate else '✓ OK'))
    )
    
    vif_df['recommendation'] = vif_df['VIF'].apply(
        lambda x: 'MUST REMOVE' if np.isinf(x)
        else ('Consider removing' if x > vif_threshold_high
        else ('Monitor' if x > vif_threshold_moderate else 'Keep'))
    )
    
    # Summary
    n_high = (vif_df['VIF'] > vif_threshold_high).sum()
    n_moderate = ((vif_df['VIF'] > vif_threshold_moderate) & (vif_df['VIF'] <= vif_threshold_high)).sum()
    
    print(f"\nVIF Summary:")
    print(f"  High VIF (>{vif_threshold_high}): {n_high}")
    print(f"  Moderate VIF ({vif_threshold_moderate}-{vif_threshold_high}): {n_moderate}")
    print(f"  Acceptable VIF (<{vif_threshold_moderate}): {len(vif_df) - n_high - n_moderate}")
    
    return vif_df.sort_values('VIF', ascending=False)


def iterative_vif_removal(
    df: pd.DataFrame,
    feature_cols: List[str],
    vif_threshold: float = 10.0,
    max_iterations: int = 50,
    verbose: bool = True,
) -> Tuple[List[str], List[Tuple[str, float]], pd.DataFrame]:
    """
    Iteratively remove features with highest VIF until all are below threshold.
    
    This is the recommended approach for handling multicollinearity:
    1. Calculate VIF for all features
    2. Remove the feature with highest VIF (if above threshold)
    3. Recalculate VIF (removing one feature often improves others)
    4. Repeat until all VIF < threshold
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    feature_cols : List[str]
        Starting feature list
    vif_threshold : float
        Remove features until all VIF <= this value
    max_iterations : int
        Safety limit on iterations
    verbose : bool
        Print progress
    
    Returns:
    --------
    Tuple of:
        - cleaned_features: List of features after removal
        - removed_features: List of (feature, vif) tuples that were removed
        - log_df: DataFrame showing iteration history
    """
    current_features = feature_cols.copy()
    removed_features = []
    iteration_log = []
    
    X = df[feature_cols].dropna()
    
    if verbose:
        print(f"\nStarting iterative VIF removal (threshold={vif_threshold})")
        print(f"Initial features: {len(current_features)}")
    
    for iteration in range(max_iterations):
        if len(current_features) <= 1:
            break
        
        # Calculate VIF
        X_current = X[current_features]
        vif_values = []
        for i, col in enumerate(current_features):
            try:
                vif = variance_inflation_factor(X_current.values, i)
            except:
                vif = np.inf
            vif_values.append((col, vif))
        
        max_feature, max_vif = max(vif_values, key=lambda x: x[1])
        
        iteration_log.append({
            'iteration': iteration + 1,
            'n_features': len(current_features),
            'max_vif_feature': max_feature,
            'max_vif': max_vif,
        })
        
        if max_vif <= vif_threshold:
            if verbose:
                print(f"✓ Converged: all VIF <= {vif_threshold}")
            break
        
        if verbose:
            print(f"  Iter {iteration + 1}: Remove '{max_feature}' (VIF={max_vif:.2f})")
        
        current_features.remove(max_feature)
        removed_features.append((max_feature, max_vif))
    
    if verbose:
        print(f"\nRemoved {len(removed_features)} features")
        print(f"Remaining: {len(current_features)} features")
    
    return current_features, removed_features, pd.DataFrame(iteration_log)


def analyze_feature_scaling_needs(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Analyze whether features need standardization/scaling.
    
    Guidelines:
    - Binary features (0/1): Do NOT scale - keep interpretable
    - Continuous with small range: Usually OK without scaling
    - Continuous with large range/variance: Consider scaling
    
    Note: Your modelling.py already handles this correctly:
    - Linear regression: No scaling (coefficients stay interpretable)
    - Logistic regression: StandardScaler applied (line 330-331)
    - Random Forest: No scaling needed (tree-based)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    feature_cols : List[str]
        Features to analyze
    
    Returns:
    --------
    pd.DataFrame
        Analysis of each feature's scaling needs
    """
    results = []
    
    for col in feature_cols:
        col_data = df[col].dropna()
        unique_vals = col_data.nunique()
        is_binary = set(col_data.unique()).issubset({0, 1, 0.0, 1.0})
        
        # Determine feature type
        if is_binary:
            feature_type = 'binary'
            scaling_needed = False
        elif unique_vals <= 10:
            feature_type = 'categorical/ordinal'
            scaling_needed = False
        else:
            feature_type = 'continuous'
            # Flag if large range or high variance
            scaling_needed = col_data.std() > 10 or abs(col_data.mean()) > 100
        
        results.append({
            'feature': col,
            'type': feature_type,
            'unique_values': unique_vals,
            'min': col_data.min(),
            'max': col_data.max(),
            'mean': round(col_data.mean(), 4),
            'std': round(col_data.std(), 4),
            'scaling_recommended': scaling_needed,
        })
    
    result_df = pd.DataFrame(results)
    
    # Summary
    print("\nFeature Scaling Analysis:")
    print(f"  Binary features: {(result_df['type'] == 'binary').sum()}")
    print(f"  Categorical/ordinal: {(result_df['type'] == 'categorical/ordinal').sum()}")
    print(f"  Continuous: {(result_df['type'] == 'continuous').sum()}")
    print(f"  Scaling recommended: {result_df['scaling_recommended'].sum()}")
    
    return result_df


def get_feature_removal_candidates(
    vif_df: pd.DataFrame,
    correlation_matrix: Optional[pd.DataFrame] = None,
    vif_threshold: float = 10.0,
) -> List[str]:
    """
    Get list of features that are candidates for removal based on VIF.
    
    When choosing which feature to remove from a correlated pair,
    consider keeping the one that's:
    - More interpretable
    - More directly related to your research question
    - Less noisy
    
    Parameters:
    -----------
    vif_df : pd.DataFrame
        Output from calculate_vif_detailed()
    correlation_matrix : pd.DataFrame, optional
        Correlation matrix to identify which features correlate
    vif_threshold : float
        Threshold above which to flag features
    
    Returns:
    --------
    List[str]
        Features recommended for removal
    """
    high_vif_features = vif_df[vif_df['VIF'] > vif_threshold]['feature'].tolist()
    
    print(f"\nFeatures with VIF > {vif_threshold}:")
    for feat in high_vif_features:
        vif_val = vif_df[vif_df['feature'] == feat]['VIF'].values[0]
        print(f"  - {feat}: VIF = {vif_val:.2f}")
    
    return high_vif_features


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
# REPORT GENERATION
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


def create_report(results: Dict) -> pd.DataFrame:
    """
    Create a clean, presentation-ready summary table.

    Parameters:
    -----------
    results : Dict
        Output from run_all_models()

    Returns:
    --------
    pd.DataFrame
        Formatted table for reporting
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
    Print comprehensive model summary for reporting.
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


# =============================================================================
# FEATURE INTERACTION ANALYSIS
# =============================================================================

def analyze_feature_interactions(
    results: Dict,
    df: pd.DataFrame,
    top_n: int = 10,
    sample_size: int = 500,
) -> pd.DataFrame:
    """
    Analyze feature interactions using SHAP interaction values.

    SHAP interaction values show how pairs of features work together
    to affect predictions. High interaction means the effect of one
    feature depends on another feature's value.

    Parameters:
    -----------
    results : Dict
        Output from run_all_models() with SHAP analysis completed
    df : pd.DataFrame
        DataFrame with features
    top_n : int
        Number of top interactions to return
    sample_size : int
        Sample size for interaction calculation (can be slow)

    Returns:
    --------
    pd.DataFrame
        Top feature interactions with strength
    """
    print("\n" + "="*70)
    print("FEATURE INTERACTION ANALYSIS")
    print("="*70)

    if 'rf_model' not in results:
        print("Run run_all_models() first")
        return pd.DataFrame()

    rf_model = results['rf_model']
    feature_cols = results['feature_cols']

    # Prepare data
    X = df[feature_cols].copy()
    mask = X.notna().all(axis=1)
    X = X[mask].reset_index(drop=True)

    # Sample for speed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
        print(f"Using {sample_size} samples for interaction analysis")
    else:
        X_sample = X

    # Calculate SHAP interaction values
    print("Calculating SHAP interaction values (this may take a moment)...")
    explainer = shap.TreeExplainer(rf_model)

    try:
        shap_interaction = explainer.shap_interaction_values(X_sample)
        print("Done!")
    except Exception as e:
        print(f"Interaction analysis failed: {e}")
        return pd.DataFrame()

    # Extract interaction strengths
    n_features = len(feature_cols)
    interactions = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Mean absolute interaction
            interaction_strength = np.abs(shap_interaction[:, i, j]).mean()
            interactions.append({
                'feature_1': feature_cols[i],
                'feature_2': feature_cols[j],
                'interaction_strength': interaction_strength,
            })

    interaction_df = pd.DataFrame(interactions)
    interaction_df = interaction_df.sort_values('interaction_strength', ascending=False)
    interaction_df['rank'] = range(1, len(interaction_df) + 1)

    # Add interpretation
    max_strength = interaction_df['interaction_strength'].max()
    interaction_df['relative_strength'] = (
        interaction_df['interaction_strength'] / max_strength * 100
    ).round(1)
    interaction_df['interpretation'] = interaction_df['relative_strength'].apply(
        lambda x: 'Strong' if x > 50 else ('Moderate' if x > 20 else 'Weak')
    )

    print(f"\nTop {top_n} Feature Interactions:")
    print(interaction_df.head(top_n)[
        ['feature_1', 'feature_2', 'interaction_strength', 'interpretation']
    ].to_string(index=False))

    # Store in results
    results['shap_interactions'] = interaction_df
    results['shap_interaction_values'] = shap_interaction

    return interaction_df.head(top_n)


# =============================================================================
# MODEL SCORECARD
# =============================================================================

def create_model_scorecard(results: Dict) -> pd.DataFrame:
    """
    Create a unified scorecard comparing all models.

    This provides a quick comparison of model performance,
    interpretability, and recommended use cases.

    Parameters:
    -----------
    results : Dict
        Output from run_all_models()

    Returns:
    --------
    pd.DataFrame
        Model comparison scorecard
    """
    print("\n" + "="*70)
    print("MODEL SCORECARD")
    print("="*70)

    scorecard = []

    # Linear Regression
    if 'linear_model' in results:
        lm = results['linear_model']
        n_significant = (results['linear_importance']['p_value'] < 0.05).sum()
        n_features = len(results['linear_importance'])

        scorecard.append({
            'Model': 'Linear Regression (OLS)',
            'Primary Metric': f"R² = {lm.rsquared:.3f}",
            'Secondary Metric': f"Adj R² = {lm.rsquared_adj:.3f}",
            'Significant Features': f"{n_significant}/{n_features}",
            'Interpretability': 'High',
            'Best For': 'Effect direction & magnitude',
            'Handles Non-Linear': 'No',
        })

    # Random Forest
    if 'rf_cv_results' in results:
        rf_cv = results['rf_cv_results']
        cv_mean = np.mean(rf_cv['test_r2'])
        cv_std = np.std(rf_cv['test_r2'])
        rmse = np.sqrt(np.mean(rf_cv['test_mse']))

        scorecard.append({
            'Model': 'Random Forest',
            'Primary Metric': f"CV R² = {cv_mean:.3f} ± {cv_std:.3f}",
            'Secondary Metric': f"RMSE = {rmse:.3f}",
            'Significant Features': 'N/A (uses importance)',
            'Interpretability': 'Medium (SHAP)',
            'Best For': 'Non-linear patterns',
            'Handles Non-Linear': 'Yes',
        })

    # Logistic Regression
    if 'lr_cv_results' in results:
        lr_cv = results['lr_cv_results']
        acc_mean = np.mean(lr_cv['accuracy'])
        acc_std = np.std(lr_cv['accuracy'])
        auc_mean = np.nanmean(lr_cv['auc'])
        auc_std = np.nanstd(lr_cv['auc'])
        n_significant = results['lr_importance']['lr_significant'].sum() if 'lr_importance' in results else 'N/A'

        scorecard.append({
            'Model': 'Logistic Regression',
            'Primary Metric': f"AUC = {auc_mean:.3f} ± {auc_std:.3f}",
            'Secondary Metric': f"Accuracy = {acc_mean:.3f} ± {acc_std:.3f}",
            'Significant Features': str(n_significant),
            'Interpretability': 'High (odds ratios)',
            'Best For': 'Risk factors (binary)',
            'Handles Non-Linear': 'No',
        })

    scorecard_df = pd.DataFrame(scorecard)

    # Print formatted scorecard
    print("\n" + scorecard_df.to_string(index=False))

    # Add recommendations
    print("\n" + "-"*70)
    print("RECOMMENDATIONS:")
    print("-"*70)

    if 'linear_model' in results and 'rf_cv_results' in results:
        linear_r2 = results['linear_model'].rsquared
        rf_r2 = np.mean(results['rf_cv_results']['test_r2'])

        if rf_r2 > linear_r2 + 0.05:
            print("  Random Forest outperforms Linear by >5% R²")
            print("  → Non-linear relationships likely exist in your data")
        elif linear_r2 > rf_r2:
            print("  Linear performs as well as Random Forest")
            print("  → Prefer Linear for interpretability")
        else:
            print("  Models perform similarly")
            print("  → Use Linear for interpretability, RF for prediction")

    if 'lr_cv_results' in results:
        auc = np.nanmean(results['lr_cv_results']['auc'])
        if auc >= 0.8:
            print(f"  Logistic Regression AUC = {auc:.3f} (Good discrimination)")
        elif auc >= 0.7:
            print(f"  Logistic Regression AUC = {auc:.3f} (Acceptable)")
        else:
            print(f"  Logistic Regression AUC = {auc:.3f} (Poor - consider other models)")

    return scorecard_df


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def analyze_logistic_calibration(
    results: Dict,
    df: pd.DataFrame,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, Any]:
    """
    Analyze calibration of logistic regression predictions.

    Calibration shows whether predicted probabilities match actual
    frequencies. A well-calibrated model predicting 30% probability
    should be correct ~30% of the time.

    Parameters:
    -----------
    results : Dict
        Output from run_all_models()
    df : pd.DataFrame
        DataFrame with features and outcome
    n_bins : int
        Number of bins for calibration analysis

    Returns:
    --------
    Tuple of (calibration_df, plotly_figure)
    """
    import plotly.graph_objects as go

    print("\n" + "="*70)
    print("LOGISTIC REGRESSION CALIBRATION")
    print("="*70)

    if 'lr_model' not in results:
        print("No logistic regression model found")
        return pd.DataFrame(), None

    lr_model = results['lr_model']
    feature_cols = results['feature_cols']

    # Prepare data
    X = df[feature_cols].copy()
    y_raw = df['wonky_study_count'].copy()
    y = (y_raw.fillna(0) > 0).astype(int)

    # Handle missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Scale features
    if hasattr(lr_model, 'scaler_'):
        X_scaled = pd.DataFrame(
            lr_model.scaler_.transform(X),
            columns=feature_cols
        )
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    # Get predicted probabilities
    y_prob = lr_model.predict_proba(X_scaled)[:, 1]

    # Create calibration bins
    calibration_data = pd.DataFrame({
        'y_true': y,
        'y_prob': y_prob,
    })

    calibration_data['bin'] = pd.cut(
        calibration_data['y_prob'],
        bins=n_bins,
        labels=False
    )

    # Calculate bin statistics
    calibration_stats = calibration_data.groupby('bin').agg({
        'y_prob': ['mean', 'count'],
        'y_true': 'mean',
    }).reset_index()

    calibration_stats.columns = ['bin', 'mean_predicted', 'n_samples', 'actual_rate']

    # Calculate calibration error
    calibration_stats['calibration_error'] = (
        calibration_stats['actual_rate'] - calibration_stats['mean_predicted']
    ).abs()

    # Overall metrics
    mean_calibration_error = calibration_stats['calibration_error'].mean()
    max_calibration_error = calibration_stats['calibration_error'].max()

    print(f"\nCalibration Summary:")
    print(f"  Mean Calibration Error: {mean_calibration_error:.4f}")
    print(f"  Max Calibration Error:  {max_calibration_error:.4f}")

    if mean_calibration_error < 0.05:
        print("  Well-calibrated (error < 5%)")
    elif mean_calibration_error < 0.10:
        print("  Acceptably calibrated (error < 10%)")
    else:
        print("  Poorly calibrated - predictions may be overconfident")

    # Create Plotly calibration plot
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='gray'),
    ))

    # Actual calibration
    fig.add_trace(go.Scatter(
        x=calibration_stats['mean_predicted'],
        y=calibration_stats['actual_rate'],
        mode='lines+markers',
        name='Model Calibration',
        marker=dict(size=10),
        line=dict(color='blue'),
        hovertemplate=(
            'Predicted: %{x:.2%}<br>'
            'Actual: %{y:.2%}<br>'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title='Logistic Regression Calibration Plot',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Actual Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True,
        height=500,
        width=600,
    )

    # Add annotation
    fig.add_annotation(
        x=0.95,
        y=0.05,
        text=f"Mean Error: {mean_calibration_error:.3f}",
        showarrow=False,
        xanchor='right',
    )

    # Store results
    results['calibration_data'] = calibration_stats
    results['calibration_error'] = mean_calibration_error

    return calibration_stats, fig


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

def generate_modelling_executive_summary(
    results: Dict,
    outcome_description: str = "wonkiness",
) -> str:
    """
    Generate a comprehensive executive summary for reporting.

    This provides a narrative summary of modelling results including
    key predictors, risk factors, and model recommendations.

    Parameters:
    -----------
    results : Dict
        Output from run_all_models()
    outcome_description : str
        Description of the outcome variable for narrative

    Returns:
    --------
    str
        Formatted executive summary
    """
    lines = []
    lines.append("="*80)
    lines.append("EXECUTIVE SUMMARY: PREDICTIVE MODELLING RESULTS")
    lines.append("="*80)

    # Model Performance Summary
    lines.append("\n1. MODEL PERFORMANCE")
    lines.append("-"*40)

    if 'linear_model' in results:
        lm = results['linear_model']
        lines.append(f"   Linear Regression:  R² = {lm.rsquared:.3f}")
        lines.append(f"      Explains {lm.rsquared*100:.1f}% of variance in {outcome_description}")

    if 'rf_cv_results' in results:
        rf_cv = results['rf_cv_results']
        rf_r2 = np.mean(rf_cv['test_r2'])
        rf_std = np.std(rf_cv['test_r2'])
        lines.append(f"   Random Forest:      R² = {rf_r2:.3f} ± {rf_std:.3f}")

    if 'lr_cv_results' in results:
        lr_cv = results['lr_cv_results']
        auc = np.nanmean(lr_cv['auc'])
        lines.append(f"   Logistic Regression: AUC = {auc:.3f}")
        if auc >= 0.8:
            lines.append("      (Good ability to distinguish wonky vs non-wonky)")
        elif auc >= 0.7:
            lines.append("      (Acceptable discrimination)")

    # Top Predictors
    lines.append("\n2. TOP PREDICTIVE FACTORS")
    lines.append("-"*40)

    comp = results['comparison']
    for i, (_, row) in enumerate(comp.head(5).iterrows(), 1):
        feat = row['feature']
        direction = ""
        if 'shap_direction' in row and pd.notna(row['shap_direction']):
            direction = f" ({row['shap_direction']} {outcome_description})"
        elif 'linear_coef' in row:
            direction = f" ({'increases' if row['linear_coef'] > 0 else 'decreases'} {outcome_description})"
        lines.append(f"   {i}. {feat}{direction}")

    # Risk Factors vs Protective Factors
    if 'shap_importance' in results:
        shap_df = results['shap_importance']

        lines.append("\n3. RISK FACTORS (increase " + outcome_description + ")")
        lines.append("-"*40)
        risk_factors = shap_df[shap_df['shap_mean'] > 0.001].head(5)
        if len(risk_factors) > 0:
            for _, row in risk_factors.iterrows():
                lines.append(f"   {row['feature']}: +{row['shap_mean']:.4f}")
        else:
            lines.append("   No clear risk factors identified")

        lines.append("\n4. PROTECTIVE FACTORS (decrease " + outcome_description + ")")
        lines.append("-"*40)
        protective = shap_df[shap_df['shap_mean'] < -0.001].head(5)
        if len(protective) > 0:
            for _, row in protective.iterrows():
                lines.append(f"   {row['feature']}: {row['shap_mean']:.4f}")
        else:
            lines.append("   No clear protective factors identified")

    # Odds Ratios (if logistic regression)
    if 'lr_importance' in results:
        lr_df = results['lr_importance'].copy()
        valid_or = lr_df[(lr_df['lr_odds_ratio'] >= 0.1) & (lr_df['lr_odds_ratio'] <= 10)]

        if len(valid_or) > 0:
            lines.append("\n5. ODDS RATIOS (Logistic Regression)")
            lines.append("-"*40)

            # Highest risk
            high_risk = valid_or[valid_or['lr_odds_ratio'] > 1.2].nlargest(3, 'lr_odds_ratio')
            if len(high_risk) > 0:
                lines.append("   Highest Risk:")
                for _, row in high_risk.iterrows():
                    pct = (row['lr_odds_ratio'] - 1) * 100
                    lines.append(f"      {row['feature']}: OR = {row['lr_odds_ratio']:.2f} ({pct:.0f}% more likely)")

            # Lowest risk
            low_risk = valid_or[valid_or['lr_odds_ratio'] < 0.8].nsmallest(3, 'lr_odds_ratio')
            if len(low_risk) > 0:
                lines.append("   Lowest Risk:")
                for _, row in low_risk.iterrows():
                    pct = (1 - row['lr_odds_ratio']) * 100
                    lines.append(f"      {row['feature']}: OR = {row['lr_odds_ratio']:.2f} ({pct:.0f}% less likely)")

    # Model Agreement
    lines.append("\n6. MODEL AGREEMENT")
    lines.append("-"*40)

    if 'linear_rank' in comp.columns and 'rf_rank' in comp.columns:
        # Check if top features agree across models
        top_linear = set(comp.nsmallest(5, 'linear_rank')['feature'])
        top_rf = set(comp.nsmallest(5, 'rf_rank')['feature'])
        agreement = top_linear & top_rf

        lines.append(f"   Features in top 5 for both Linear & RF: {len(agreement)}")
        if agreement:
            lines.append(f"      {', '.join(sorted(agreement))}")

    # Key Takeaways
    lines.append("\n7. KEY TAKEAWAYS")
    lines.append("-"*40)

    # Calculate some key insights
    n_significant = (results['linear_importance']['p_value'] < 0.05).sum() if 'linear_importance' in results else 0
    n_features = len(results['feature_cols'])

    lines.append(f"   {n_significant} of {n_features} features significantly predict {outcome_description}")

    if 'linear_model' in results:
        r2 = results['linear_model'].rsquared
        if r2 >= 0.5:
            lines.append(f"   Strong explanatory power (R² = {r2:.2f})")
        elif r2 >= 0.3:
            lines.append(f"   Moderate explanatory power (R² = {r2:.2f})")
        else:
            lines.append(f"   Low explanatory power (R² = {r2:.2f}) - other factors may be involved")

    lines.append("\n" + "="*80)

    summary = "\n".join(lines)
    print(summary)

    return summary