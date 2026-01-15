"""
Visualizations Module

Helper functions for creating plotly visualizations.
"""

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Optional, Dict, List


def create_breakdown_summary(
    df: pd.DataFrame,
    features: List[str],
    group_col: str = "wonky_study_count",
    group_threshold: float = 0,
) -> str:
    """
    Create formatted text summary showing percentage breakdown for selected features.
    """
    if group_col not in df.columns:
        return f"Group column '{group_col}' not found in DataFrame."

    summary_lines = ["features created:"]

    # Precompute masks once
    wonky_mask = df[group_col] > group_threshold
    has_nan = df[group_col].isna().any()
    non_wonky_mask = df[group_col].isna() if has_nan else (df[group_col] == 0)
    non_wonky_condition = f"{group_col} is NaN" if has_nan else f"{group_col} = 0"

    for feature in features:
        print(f'working on {feature}')
        if feature not in df.columns:
            continue

        all_pct = df[feature].mean() * 100
        wonky_pct = df.loc[wonky_mask, feature].mean() * 100 if wonky_mask.any() else 0.0
        non_wonky_pct = (
            df.loc[non_wonky_mask, feature].mean() * 100
            if non_wonky_mask.any()
            else 0.0
        )
        delta_pct = wonky_pct - non_wonky_pct

        feature_display = feature.replace("_", " ").title()

        summary_lines.append(f"  - {feature_display}: {all_pct:.1f}%")
        summary_lines.append(f"    * All tasks: {all_pct:.1f}%")
        summary_lines.append(
            f"    * Wonky study tasks ({group_col} > {group_threshold}): {wonky_pct:.1f}%"
        )
        summary_lines.append(
            f"    * Non-wonky study tasks ({non_wonky_condition}): {non_wonky_pct:.1f}%"
        )
        summary_lines.append(
            f"    * Delta (wonky - non-wonky): {delta_pct:+.1f}%"
        )

    return "\n".join(summary_lines)


def create_task_speed_breakdown_summary(
    df: pd.DataFrame,
    group_col: str = "wonky_study_count",
    group_threshold: float = 0
) -> str:
    """
    Create formatted text summary showing percentages for task speed features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with speed features and group column
    group_col : str
        Column name for grouping (default: "wonky_study_count")
    group_threshold : float
        Threshold for determining wonky vs non-wonky groups
        
    Returns:
    --------
    str
        Formatted summary string
    """
    summary_lines = ["Task speed features breakdown:"]
    
    # Check if group column exists
    if group_col not in df.columns:
        available_cols = [col for col in df.columns if 'wonky' in col.lower() or 'study' in col.lower()]
        error_msg = (
            f"Group column '{group_col}' not found in DataFrame.\n"
            f"Available columns with 'wonky' or 'study': {available_cols[:10] if available_cols else 'None'}\n"
            f"All columns: {list(df.columns)[:20]}"
        )
        return error_msg
    
    speed_features = ['is_suspiciously_fast', 'is_fast', 'is_normal_speed', 'is_slow', 'is_suspiciously_slow']
    
    for feature in speed_features:
        if feature not in df.columns:
            continue
        
        all_pct = df[feature].mean() * 100
        
        wonky_mask = df[group_col] > group_threshold
        
        if df[group_col].isna().sum() > 0:
            non_wonky_mask = df[group_col].isna()
        else:
            non_wonky_mask = df[group_col] == 0
        
        if wonky_mask.sum() > 0:
            wonky_pct = df.loc[wonky_mask, feature].mean() * 100
        else:
            wonky_pct = 0.0
        
        if non_wonky_mask.sum() > 0:
            non_wonky_pct = df.loc[non_wonky_mask, feature].mean() * 100
        else:
            non_wonky_pct = 0.0
        
        delta_pct = wonky_pct - non_wonky_pct
        
        if df[group_col].isna().sum() > 0:
            non_wonky_condition = f"{group_col} is NaN"
        else:
            non_wonky_condition = f"{group_col} = 0"
        
        feature_display = feature.replace('_', ' ').title()
        if feature == 'is_fast':
            feature_display = "Fast tasks (1 std dev below mean)"
        elif feature == 'is_suspiciously_fast':
            feature_display = "Suspiciously fast tasks (2 std dev below mean)"
        elif feature == 'is_normal_speed':
            feature_display = "Normal speed tasks (within 1 std dev of mean)"
        elif feature == 'is_slow':
            feature_display = "Slow tasks (1 std dev above mean)"
        elif feature == 'is_suspiciously_slow':
            feature_display = "Suspiciously slow tasks (2 std dev above mean)"
        
        summary_lines.append(f"  - {feature_display}: {all_pct:.1f}%")
        summary_lines.append(f"    * All tasks: {all_pct:.1f}%")
        summary_lines.append(f"    * Wonky study tasks ({group_col} > {group_threshold}): {wonky_pct:.1f}%")
        summary_lines.append(f"    * Non-wonky study tasks ({non_wonky_condition}): {non_wonky_pct:.1f}%")
        summary_lines.append(f"    * Delta (wonky - non-wonky): {delta_pct:+.1f}%")
    
    return "\n".join(summary_lines)


def create_chi_squared_bar_chart(
    test_results_df: pd.DataFrame,
    chi2_col: str = "chi2",
    p_value_col: str = "chi_p_value",
    significance_level: float = 0.01,
    title: str = "Chi-Squared Statistic by Temporal Feature",
    xaxis_title: str = "Temporal Feature",
    yaxis_title: str = "Chi-Squared Statistic"
) -> go.Figure:
    """
    Create bar chart of chi-squared statistics by temporal feature.
    
    Parameters:
    -----------
    test_results_df : pd.DataFrame
        DataFrame with temporal features as index and chi2/p_value columns
    chi2_col : str
        Column name for chi-squared statistic
    p_value_col : str
        Column name for p-value
    significance_level : float
        Significance level for coloring bars
    title : str
        Chart title
    xaxis_title : str
        X-axis title
    yaxis_title : str
        Y-axis title
        
    Returns:
    --------
    go.Figure
        Plotly figure with chi-squared bar chart
    """

    if test_results_df.index.name is None:
        features = test_results_df.index.tolist()
    else:
        features = test_results_df.index.tolist()
    
    chi2_values = test_results_df[chi2_col].values
    p_values = test_results_df[p_value_col].values if p_value_col in test_results_df.columns else None
    
    # Color bars by significance if p-values available
    if p_values is not None:
        colors = ['indianred' if p < significance_level else 'lightcoral' for p in p_values]
    else:
        colors = 'indianred'
    
    fig = go.Figure(go.Bar(
        x=features,
        y=chi2_values,
        text=[f"{x:.2f}" for x in chi2_values],
        textposition="inside",
        marker_color=colors
    ))
    
    fig.update_layout(
        title=title,
        yaxis=dict(title=yaxis_title),
        xaxis=dict(title=xaxis_title)
    )
    
    return fig


def calculate_temporal_feature_deltas(
    df: pd.DataFrame,
    temporal_features: List[str],
    group_col: str = "wonky_study_count",
    group_threshold: float = 0
) -> pd.DataFrame:
    """
    Calculate delta percentages (wonky - non-wonky) for temporal features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with temporal features and group column
    temporal_features : List[str]
        List of temporal feature column names
    group_col : str
        Column name for grouping (default: "wonky_study_count")
    group_threshold : float
        Threshold for determining wonky vs non-wonky groups
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features as index and 'delta_pct' column
    """
    if group_col not in df.columns:
        return pd.DataFrame()
    
    deltas = []
    feature_names = []
    
    for feature in temporal_features:
        if feature not in df.columns:
            continue
        
        wonky_mask = df[group_col] > group_threshold
        
        if df[group_col].isna().sum() > 0:
            non_wonky_mask = df[group_col].isna()
        else:
            non_wonky_mask = df[group_col] == 0
        
        if wonky_mask.sum() > 0:
            wonky_pct = df.loc[wonky_mask, feature].mean() * 100
        else:
            wonky_pct = 0.0
        
        if non_wonky_mask.sum() > 0:
            non_wonky_pct = df.loc[non_wonky_mask, feature].mean() * 100
        else:
            non_wonky_pct = 0.0
        
        delta_pct = wonky_pct - non_wonky_pct
        
        deltas.append(delta_pct)
        feature_names.append(feature)
    
    result_df = pd.DataFrame({
        'delta_pct': deltas
    }, index=feature_names)
    
    return result_df


def create_feature_breakdown_table(
    df: pd.DataFrame,
    feature_col: str,
    group_col: str,
    group1_value: int = 1,
    group2_value: int = 0,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create summary table showing wonky vs non-wonky breakdowns by feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with feature and group columns
    feature_col : str
        Column name for the feature to analyze
    group_col : str
        Column name for group labels
    group1_value : int
        Value for first group (e.g., 1 for wonky)
    group2_value : int
        Value for second group (e.g., 0 for non-wonky)
    metrics : List[str], optional
        List of metric columns to summarize. If None, uses numeric columns.
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with breakdown statistics
    """
    if feature_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    
    group1 = df[df[group_col] == group1_value]
    group2 = df[df[group_col] == group2_value]
    
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        metrics = [m for m in metrics if m not in [feature_col, group_col]]
    
    summary_data = []
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        group1_values = group1[metric].dropna()
        group2_values = group2[metric].dropna()
        
        if len(group1_values) == 0 or len(group2_values) == 0:
            continue
        
        summary_data.append({
            'metric': metric,
            'wonky_mean': group1_values.mean(),
            'wonky_median': group1_values.median(),
            'wonky_std': group1_values.std(),
            'wonky_count': len(group1_values),
            'non_wonky_mean': group2_values.mean(),
            'non_wonky_median': group2_values.median(),
            'non_wonky_std': group2_values.std(),
            'non_wonky_count': len(group2_values),
            'mean_difference': group1_values.mean() - group2_values.mean(),
            'median_difference': group1_values.median() - group2_values.median()
        })
    
    return pd.DataFrame(summary_data)


def create_breakdown_chart(
    df: pd.DataFrame,
    features: List[str],
    group_col: str = "wonky_study_count",
    group_threshold: float = 0,
) -> go.Figure:
    """
    Create Plotly chart showing percentage delta between two groups for selected features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with selected features and group column
    features : List[str]
        List of features column names
    group_col : str, default "wonky_study_count"
        Column name for grouping
    group_threshold : float, default 0
        Threshold for determining wonky vs non-wonky groups

    Returns
    -------
    go.Figure
        Plotly bar chart showing deltas
    """
    if group_col not in df.columns:
        available_cols = [
            col for col in df.columns
            if 'wonky' in col.lower() or 'study' in col.lower()
        ]
        raise ValueError(
            f"Group column '{group_col}' not found. "
            f"Available: {available_cols[:10] if available_cols else 'None'}"
        )

    # Specific for temporal
    feature_display_map = {
        'is_weekend': "Weekend tasks",
        'is_night': "Night tasks (10 PM - 6 AM)",
        'is_business_hour': "Business hour tasks (9 AM - 5 PM)",
        'is_business_hour_weekday': "Business hour tasks weekday",
        'is_business_hour_weekend': "Business hour tasks weekend",
    }

    features_data = []

    for feature in features:
        if feature not in df.columns:
            continue

        wonky_mask = df[group_col] > group_threshold
        
        if df[group_col].isna().sum() > 0:
            non_wonky_mask = df[group_col].isna()
        else:
            non_wonky_mask = df[group_col] == 0

        wonky_pct = (
            df.loc[wonky_mask, feature].mean() * 100
            if wonky_mask.sum() > 0 else 0.0
        )
        non_wonky_pct = (
            df.loc[non_wonky_mask, feature].mean() * 100
            if non_wonky_mask.sum() > 0 else 0.0
        )
        
        delta_pct = wonky_pct - non_wonky_pct

        display_name = feature_display_map.get(
            feature,
            feature.replace('_', ' ').title()
        )

        features_data.append({
            'feature': display_name,
            'delta': delta_pct,
            'wonky_pct': wonky_pct,
            'non_wonky_pct': non_wonky_pct,
        })

    chart_df = pd.DataFrame(features_data)
    chart_df = chart_df.sort_values('delta', ascending=True)

    colors = ['#ff4b4b' if x < 0 else '#51cf66' for x in chart_df['delta']]

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_df['delta'],
                y=chart_df['feature'],
                orientation='h',
                marker_color=colors,
                text=[f"{x:+.1f}%" for x in chart_df['delta']],
                textposition='auto',
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Delta: %{x:+.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Feature Differences: Wonky vs Non-Wonky Groups",
        xaxis_title="% Point Delta (Wonky - Non-Wonky)",
        yaxis_title="Feature",
        height=400 + (len(chart_df) * 20),
        showlegend=False,
        hovermode='closest',
        template='plotly_white',
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def compute_exposure_and_gender_counts(
    wonky_respondent_summary: pd.DataFrame,
    user_info_df: pd.DataFrame,
    labels: list,
) -> pd.DataFrame:
    """
    Merge respondent summary with user info, reshape to long,
    and compute counts and percentages of respondents by
    labels, exposure band, and gender.
    """

    wonky_tasknum = wonky_respondent_summary.merge(
        user_info_df[task_labels + ["respondentPk", "gender"]],
        on="respondentPk",
        how="left",
    )

    long = (
        wonky_tasknum[["respondentPk", "exposure_band", "gender"] + labels]
        .melt(
            id_vars=["respondentPk", "exposure_band", "gender"],
            value_vars=labels,
            var_name="label_tags",
            value_name="flag",
        )
    )

    long = long[long["flag"] == 1]

    counts = (
        long
        .groupby(["label_tags", "exposure_band", "gender"])
        .agg(n=("respondentPk", "nunique"))
        .reset_index()
    )

    counts["group_total"] = (
        counts
        .groupby(["exposure_band", "gender"])["n"]
        .transform("sum")
    )

    counts["pct"] = counts["n"] / counts["group_total"] * 100

    return counts


# =============================================================================
# STAKEHOLDER-FRIENDLY VISUALIZATION FUNCTIONS
# =============================================================================

def create_coefficient_forest_plot(
    results_df: pd.DataFrame,
    coefficient_col: str = 'mean_difference',
    ci_lower_col: str = 'ci_lower',
    ci_upper_col: str = 'ci_upper',
    p_value_col: str = 'p_value',
    feature_col: str = 'feature',
    title: str = "Feature Effects with 95% Confidence Intervals",
    max_features: int = 20,
    significance_level: float = 0.05,
) -> go.Figure:
    """
    Create forest plot showing coefficients with confidence intervals.

    Visual elements:
    - Horizontal bars for point estimates
    - Error bars for confidence intervals
    - Color coding: green=protective, red=risk, gray=not significant
    - Vertical line at 0 (null effect)

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with coefficient, CI, and p-value columns
    coefficient_col : str
        Column name for coefficient values
    ci_lower_col : str
        Column name for CI lower bound
    ci_upper_col : str
        Column name for CI upper bound
    p_value_col : str
        Column name for p-values
    feature_col : str
        Column name for feature names
    title : str
        Chart title
    max_features : int
        Maximum number of features to display
    significance_level : float
        P-value threshold for significance coloring

    Returns
    -------
    go.Figure
        Plotly figure
    """
    df = results_df.reset_index() if feature_col not in results_df.columns else results_df.copy()

    # Sort by absolute effect size and take top N
    df['abs_effect'] = df[coefficient_col].abs()
    df = df.nlargest(max_features, 'abs_effect')
    df = df.sort_values(coefficient_col)

    # Color by direction and significance
    colors = []
    for _, row in df.iterrows():
        if row[p_value_col] >= significance_level:
            colors.append('#888888')  # Gray for non-significant
        elif row[coefficient_col] > 0:
            colors.append('#e74c3c')  # Red for increases risk
        else:
            colors.append('#27ae60')  # Green for decreases risk

    # Calculate error bar values
    error_plus = df[ci_upper_col] - df[coefficient_col]
    error_minus = df[coefficient_col] - df[ci_lower_col]

    fig = go.Figure()

    # Add points with error bars
    fig.add_trace(go.Scatter(
        x=df[coefficient_col],
        y=df[feature_col],
        mode='markers',
        marker=dict(size=10, color=colors),
        error_x=dict(
            type='data',
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color='rgba(0,0,0,0.3)',
            thickness=1.5,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Effect: %{x:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    # Add vertical line at 0 (null effect)
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)

    # Add annotations
    fig.add_annotation(
        x=0.02,
        y=1.05,
        xref="paper",
        yref="paper",
        text="<b>Green</b> = Protective (decreases risk) | <b>Red</b> = Risk Factor (increases risk) | <b>Gray</b> = Not Significant",
        showarrow=False,
        font=dict(size=10),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Effect Size (Coefficient)",
        yaxis_title="Feature",
        height=400 + len(df) * 25,
        showlegend=False,
        template='plotly_white',
        margin=dict(l=200, t=80),
    )

    return fig


def create_odds_ratio_plot(
    results_df: pd.DataFrame,
    or_col: str = 'odds_ratio',
    ci_lower_col: str = 'or_ci_lower',
    ci_upper_col: str = 'or_ci_upper',
    p_value_col: str = 'p_value',
    feature_col: str = 'feature',
    title: str = "Odds Ratios with 95% Confidence Intervals",
    max_features: int = 20,
    significance_level: float = 0.05,
) -> go.Figure:
    """
    Create forest plot for odds ratios (log scale).

    Visual elements:
    - Log scale x-axis (OR=1 is neutral)
    - Vertical line at OR=1
    - Color coding: green=protective (OR<1), red=risk (OR>1), gray=not significant

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with OR, CI, and p-value columns
    or_col : str
        Column name for odds ratio values
    ci_lower_col : str
        Column name for CI lower bound
    ci_upper_col : str
        Column name for CI upper bound
    p_value_col : str
        Column name for p-values
    feature_col : str
        Column name for feature names
    title : str
        Chart title
    max_features : int
        Maximum number of features to display
    significance_level : float
        P-value threshold for significance coloring

    Returns
    -------
    go.Figure
        Plotly figure
    """
    df = results_df.reset_index() if feature_col not in results_df.columns else results_df.copy()

    # Filter valid ORs
    df = df[(df[or_col] > 0.01) & (df[or_col] < 100)].copy()

    # Sort by distance from 1 (log scale) and take top N
    df['log_or'] = np.log(df[or_col])
    df['abs_log_or'] = df['log_or'].abs()
    df = df.nlargest(max_features, 'abs_log_or')
    df = df.sort_values('log_or')

    # Color by direction and significance
    colors = []
    for _, row in df.iterrows():
        if row[p_value_col] >= significance_level:
            colors.append('#888888')  # Gray for non-significant
        elif row[or_col] > 1:
            colors.append('#e74c3c')  # Red for increases risk
        else:
            colors.append('#27ae60')  # Green for decreases risk

    # Check if CI columns exist
    has_ci = ci_lower_col in df.columns and ci_upper_col in df.columns

    fig = go.Figure()

    if has_ci:
        # Calculate error bar values for log scale
        error_plus = np.log(df[ci_upper_col]) - np.log(df[or_col])
        error_minus = np.log(df[or_col]) - np.log(df[ci_lower_col])

        fig.add_trace(go.Scatter(
            x=df['log_or'],
            y=df[feature_col],
            mode='markers',
            marker=dict(size=10, color=colors),
            error_x=dict(
                type='data',
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                color='rgba(0,0,0,0.3)',
                thickness=1.5,
            ),
            customdata=df[or_col],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Odds Ratio: %{customdata:.2f}<br>"
                "<extra></extra>"
            ),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['log_or'],
            y=df[feature_col],
            mode='markers',
            marker=dict(size=10, color=colors),
            customdata=df[or_col],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Odds Ratio: %{customdata:.2f}<br>"
                "<extra></extra>"
            ),
        ))

    # Add vertical line at OR=1 (log(1)=0)
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)

    # Add annotations
    fig.add_annotation(
        x=0.02,
        y=1.05,
        xref="paper",
        yref="paper",
        text="<b>Green</b> = Protective (OR<1) | <b>Red</b> = Risk Factor (OR>1) | <b>Gray</b> = Not Significant",
        showarrow=False,
        font=dict(size=10),
    )

    # Custom tick labels showing actual OR values
    log_ticks = [-2, -1, -0.5, 0, 0.5, 1, 2]
    or_labels = [f"{np.exp(t):.2f}" for t in log_ticks]

    fig.update_layout(
        title=title,
        xaxis_title="Odds Ratio (log scale)",
        yaxis_title="Feature",
        height=400 + len(df) * 25,
        showlegend=False,
        template='plotly_white',
        margin=dict(l=200, t=80),
        xaxis=dict(
            tickmode='array',
            tickvals=log_ticks,
            ticktext=or_labels,
        ),
    )

    return fig


def create_model_comparison_heatmap(
    comparison_df: pd.DataFrame,
    feature_col: str = 'feature',
    rank_cols: Optional[List[str]] = None,
    max_features: int = 20,
    title: str = "Feature Importance Ranking Across Models",
) -> go.Figure:
    """
    Create heatmap showing feature ranking across different models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with feature rankings from multiple models
    feature_col : str
        Column name for feature names
    rank_cols : List[str], optional
        List of rank column names. If None, auto-detect columns ending in '_rank'
    max_features : int
        Maximum features to show
    title : str
        Chart title

    Returns
    -------
    go.Figure
        Plotly heatmap figure
    """
    df = comparison_df.copy()

    # Auto-detect rank columns if not provided
    if rank_cols is None:
        rank_cols = [c for c in df.columns if c.endswith('_rank') or c.endswith('_importance')]

    if not rank_cols:
        raise ValueError("No rank columns found. Provide rank_cols parameter.")

    # Take top N features
    if 'avg_rank' in df.columns:
        df = df.nsmallest(max_features, 'avg_rank')
    else:
        df = df.head(max_features)

    # Prepare data for heatmap
    features = df[feature_col].tolist()
    z_data = df[rank_cols].values.T

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=features,
        y=[c.replace('_rank', '').replace('_importance', '').title() for c in rank_cols],
        colorscale='RdYlGn_r',  # Reversed: green=low rank (important), red=high rank
        text=[[f"{val:.1f}" for val in row] for row in z_data],
        texttemplate="%{text}",
        hovertemplate=(
            "Feature: %{x}<br>"
            "Model: %{y}<br>"
            "Rank: %{z:.1f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Feature",
        yaxis_title="Model",
        height=200 + len(rank_cols) * 50,
        xaxis=dict(tickangle=45),
        template='plotly_white',
    )

    return fig