"""
Visualizations Module

Helper functions for creating plotly visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, List


def create_histogram(
    df: pd.DataFrame,
    x: str,
    color: Optional[str] = None,
    nbins: int = 50,
    marginal: Optional[str] = None,
    title: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    bargap: float = 0.1,
    **kwargs
) -> go.Figure:
    """
    Create a histogram plot using plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to plot
    x : str
        Column name for x-axis
    color : str, optional
        Column name for color grouping
    nbins : int
        Number of bins
    marginal : str, optional
        Type of marginal plot ('box', 'violin', 'rug', 'histogram')
    title : str, optional
        Plot title
    labels : dict, optional
        Dictionary mapping column names to display labels
    bargap : float
        Gap between bars
    **kwargs
        Additional arguments passed to px.histogram
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = px.histogram(
        df,
        x=x,
        color=color,
        nbins=nbins,
        marginal=marginal,
        title=title,
        labels=labels,
        **kwargs
    )
    
    fig.update_layout(bargap=bargap)
    
    return fig


def create_box_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    category_orders: Optional[Dict[str, List]] = None,
    boxmean: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Create a box plot using plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to plot
    x : str
        Column name for x-axis (categorical)
    y : str
        Column name for y-axis (numeric)
    color : str, optional
        Column name for color grouping
    title : str, optional
        Plot title
    labels : dict, optional
        Dictionary mapping column names to display labels
    category_orders : dict, optional
        Dictionary specifying order of categories
    boxmean : str, optional
        If 'sd', adds standard deviation to box plot
    **kwargs
        Additional arguments passed to px.box
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        labels=labels,
        category_orders=category_orders,
        **kwargs
    )
    
    if boxmean:
        fig.update_traces(boxmean=boxmean)
    
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    hover_data: Optional[List[str]] = None,
    opacity: float = 0.6,
    trendline: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Create a scatter plot using plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to plot
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    color : str, optional
        Column name for color grouping
    title : str, optional
        Plot title
    labels : dict, optional
        Dictionary mapping column names to display labels
    hover_data : list, optional
        List of column names to show on hover
    opacity : float
        Opacity of points
    trendline : str, optional
        Type of trendline ('ols' for linear regression)
    **kwargs
        Additional arguments passed to px.scatter
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        labels=labels,
        hover_data=hover_data,
        opacity=opacity,
        trendline=trendline,
        **kwargs
    )
    
    return fig


def create_bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    color_continuous_scale: Optional[str] = None,
    text: Optional[str] = None,
    texttemplate: Optional[str] = None,
    textposition: Optional[str] = None,
    tickangle: Optional[int] = None,
    **kwargs
) -> go.Figure:
    """
    Create a bar plot using plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to plot
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    color : str, optional
        Column name for color mapping
    title : str, optional
        Plot title
    labels : dict, optional
        Dictionary mapping column names to display labels
    color_continuous_scale : str, optional
        Color scale name (e.g., 'Reds', 'Blues')
    text : str, optional
        Column name for text labels on bars
    texttemplate : str, optional
        Template for text (e.g., '%{text}')
    textposition : str, optional
        Position of text ('outside', 'inside')
    tickangle : int, optional
        Angle for x-axis tick labels
    **kwargs
        Additional arguments passed to px.bar
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        labels=labels,
        color_continuous_scale=color_continuous_scale,
        text=text,
        **kwargs
    )
    
    if texttemplate:
        fig.update_traces(texttemplate=texttemplate)
    if textposition:
        fig.update_traces(textposition=textposition)
    if tickangle:
        fig.update_xaxes(tickangle=tickangle)
    
    return fig


def create_heatmap(
    pivot_data: pd.DataFrame,
    title: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    color_continuous_scale: Optional[str] = None,
    tickangle: Optional[int] = None,
    **kwargs
) -> go.Figure:
    """
    Create a heatmap using plotly.
    
    Parameters:
    -----------
    pivot_data : pd.DataFrame
        Pivot table DataFrame for heatmap
    title : str, optional
        Plot title
    labels : dict, optional
        Dictionary mapping axis names to display labels
    color_continuous_scale : str, optional
        Color scale name (e.g., 'Reds', 'Blues')
    tickangle : int, optional
        Angle for x-axis tick labels
    **kwargs
        Additional arguments passed to px.imshow
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = px.imshow(
        pivot_data,
        title=title,
        labels=labels,
        color_continuous_scale=color_continuous_scale,
        aspect='auto',
        **kwargs
    )
    
    if tickangle:
        fig.update_xaxes(tickangle=tickangle)
    
    return fig


def create_comparison_subplots(
    df: pd.DataFrame,
    metrics: List[str],
    group_col: str,
    group1_value: int = 1,
    group2_value: int = 0,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    title: str = "Comparison: Group 1 vs Group 2",
    max_plots: int = 6
) -> go.Figure:
    """
    Create subplots comparing two groups across multiple metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with metrics and group labels
    metrics : List[str]
        List of metric column names to plot
    group_col : str
        Column name containing group labels
    group1_value : int
        Value for first group
    group2_value : int
        Value for second group
    group1_name : str
        Display name for first group
    group2_name : str
        Display name for second group
    title : str
        Overall plot title
    max_plots : int
        Maximum number of plots to show
        
    Returns:
    --------
    go.Figure
        Plotly figure with subplots
    """
    available_metrics = [m for m in metrics if m in df.columns][:max_plots]
    
    if len(available_metrics) == 0:
        raise ValueError("No valid metrics found in DataFrame")
    
    n_rows = (len(available_metrics) + 1) // 2
    n_cols = 2
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=available_metrics,
        vertical_spacing=0.15
    )
    
    group1_data = df[df[group_col] == group1_value]
    group2_data = df[df[group_col] == group2_value]
    
    for idx, metric in enumerate(available_metrics):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        fig.add_trace(
            go.Box(
                y=group1_data[metric].dropna(),
                name=group1_name,
                marker_color='red',
                boxmean='sd'
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Box(
                y=group2_data[metric].dropna(),
                name=group2_name,
                marker_color='blue',
                boxmean='sd'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=title,
        height=300 * n_rows,
        showlegend=False
    )
    
    return fig


def create_temporal_breakdown_summary(
    df: pd.DataFrame,
    temporal_features: List[str],
    group_col: str = "wonky_study_count",
    group_threshold: float = 0
) -> str:
    """
    Create formatted text summary showing percentages for temporal features.
    
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
    str
        Formatted summary string
    """
    summary_lines = ["Temporal features created:"]
    
    # Check if group column exists, provide helpful error if not
    if group_col not in df.columns:
        available_cols = [col for col in df.columns if 'wonky' in col.lower() or 'study' in col.lower()]
        error_msg = (
            f"Group column '{group_col}' not found in DataFrame.\n"
            f"Available columns with 'wonky' or 'study': {available_cols[:10] if available_cols else 'None'}\n"
            f"All columns: {list(df.columns)[:20]}"
        )
        return error_msg
    
    for feature in temporal_features:
        if feature not in df.columns:
            continue
            
        # Calculate percentages
        all_pct = df[feature].mean() * 100
        
        # Wonky group (group_col > threshold)
        # Match original logic: wonky_study_count > 0
        wonky_mask = df[group_col] > group_threshold
        
        # Non-wonky group: match original logic which uses .isna()
        # Original code: user_info_df[user_info_df['wonky_study_count'].isna()]
        # Handle both cases: if column has NaN values, use .isna(); otherwise use == 0
        if df[group_col].isna().sum() > 0:
            # Has NaN values - use original logic (NaN = non-wonky)
            non_wonky_mask = df[group_col].isna()
        else:
            # No NaN values (e.g., after fillna(0)) - use == 0 for non-wonky
            # This matches the case where NaN was filled with 0
            non_wonky_mask = df[group_col] == 0
        
        # Calculate percentages for each group
        if wonky_mask.sum() > 0:
            wonky_pct = df.loc[wonky_mask, feature].mean() * 100
        else:
            wonky_pct = 0.0
        
        if non_wonky_mask.sum() > 0:
            non_wonky_pct = df.loc[non_wonky_mask, feature].mean() * 100
        else:
            non_wonky_pct = 0.0
        
        # Calculate delta (wonky - non-wonky)
        delta_pct = wonky_pct - non_wonky_pct
        
        # Determine non-wonky condition text for display
        if df[group_col].isna().sum() > 0:
            non_wonky_condition = f"{group_col} is NaN"
        else:
            non_wonky_condition = f"{group_col} = 0"
        
        # Format feature name for display
        feature_display = feature.replace('_', ' ').title()
        if feature == 'is_weekend':
            feature_display = "Weekend tasks"
        elif feature == 'is_night':
            feature_display = "Night tasks (10 PM - 6 AM)"
        elif feature == 'is_business_hour':
            feature_display = "Business hour tasks (9 AM - 5 PM)"
        elif feature == 'is_business_hour_weekday':
            feature_display = "Business hour tasks weekday"
        elif feature == 'is_business_hour_weekend':
            feature_display = "Business hour tasks weekend"
        
        summary_lines.append(f"  - {feature_display}: {all_pct:.1f}%")
        summary_lines.append(f"    * All tasks: {all_pct:.1f}%")
        summary_lines.append(f"    * Wonky study tasks ({group_col} > {group_threshold}): {wonky_pct:.1f}%")
        summary_lines.append(f"    * Non-wonky study tasks ({non_wonky_condition}): {non_wonky_pct:.1f}%")
        summary_lines.append(f"    * Delta (wonky - non-wonky): {delta_pct:+.1f}%")
    
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
    
    # Speed features to analyze (including normal speed)
    speed_features = ['is_suspiciously_fast', 'is_fast', 'is_normal_speed', 'is_slow', 'is_suspiciously_slow']
    
    for feature in speed_features:
        if feature not in df.columns:
            continue
        
        # Calculate percentages
        all_pct = df[feature].mean() * 100
        
        # Wonky group (group_col > threshold)
        wonky_mask = df[group_col] > group_threshold
        
        # Non-wonky group: handle both NaN and == 0 cases
        if df[group_col].isna().sum() > 0:
            non_wonky_mask = df[group_col].isna()
        else:
            non_wonky_mask = df[group_col] == 0
        
        # Calculate percentages for each group
        if wonky_mask.sum() > 0:
            wonky_pct = df.loc[wonky_mask, feature].mean() * 100
        else:
            wonky_pct = 0.0
        
        if non_wonky_mask.sum() > 0:
            non_wonky_pct = df.loc[non_wonky_mask, feature].mean() * 100
        else:
            non_wonky_pct = 0.0
        
        # Calculate delta (wonky - non-wonky)
        delta_pct = wonky_pct - non_wonky_pct
        
        # Determine non-wonky condition text for display
        if df[group_col].isna().sum() > 0:
            non_wonky_condition = f"{group_col} is NaN"
        else:
            non_wonky_condition = f"{group_col} = 0"
        
        # Format feature name for display
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
    # Prepare data
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
        
        # Wonky group (group_col > threshold)
        wonky_mask = df[group_col] > group_threshold
        
        # Non-wonky group: handle both NaN and == 0 cases
        if df[group_col].isna().sum() > 0:
            non_wonky_mask = df[group_col].isna()
        else:
            non_wonky_mask = df[group_col] == 0
        
        # Calculate percentages
        if wonky_mask.sum() > 0:
            wonky_pct = df.loc[wonky_mask, feature].mean() * 100
        else:
            wonky_pct = 0.0
        
        if non_wonky_mask.sum() > 0:
            non_wonky_pct = df.loc[non_wonky_mask, feature].mean() * 100
        else:
            non_wonky_pct = 0.0
        
        # Calculate delta (wonky - non-wonky)
        delta_pct = wonky_pct - non_wonky_pct
        
        deltas.append(delta_pct)
        feature_names.append(feature)
    
    result_df = pd.DataFrame({
        'delta_pct': deltas
    }, index=feature_names)
    
    return result_df


def create_chi_squared_delta_dual_axis_chart(
    chi_square_results: pd.DataFrame,
    delta_results: pd.DataFrame,
    chi2_col: str = "chi2",
    p_value_col: str = "chi_p_value",
    delta_col: str = "delta_pct",
    significance_level: float = 0.01,
    title: str = "Chi-Squared Statistic and Delta by Temporal Feature",
    xaxis_title: str = "Temporal Feature",
    y1_title: str = "Chi-Squared Statistic",
    y2_title: str = "Delta % (Wonky - Non-wonky)"
) -> go.Figure:
    """
    Create dual-axis chart: bars for chi-squared statistics, line for deltas.
    
    Parameters:
    -----------
    chi_square_results : pd.DataFrame
        DataFrame with temporal features as index and chi2/p_value columns
    delta_results : pd.DataFrame
        DataFrame with temporal features as index and delta_pct column
    chi2_col : str
        Column name for chi-squared statistic
    p_value_col : str
        Column name for p-value
    delta_col : str
        Column name for delta percentage
    significance_level : float
        Significance level for coloring bars
    title : str
        Chart title
    xaxis_title : str
        X-axis title
    y1_title : str
        Left Y-axis title (for chi-squared bars)
    y2_title : str
        Right Y-axis title (for delta line)
        
    Returns:
    --------
    go.Figure
        Plotly figure with dual-axis chart
    """
    # Get common features (intersection of both DataFrames)
    chi_features = set(chi_square_results.index.tolist())
    delta_features = set(delta_results.index.tolist())
    common_features = sorted(list(chi_features & delta_features))
    
    if len(common_features) == 0:
        # Return empty figure if no common features
        fig = go.Figure()
        fig.add_annotation(
            text="No common features between chi-squared results and delta results",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Prepare data
    chi2_values = [chi_square_results.loc[f, chi2_col] for f in common_features]
    p_values = chi_square_results.loc[common_features, p_value_col].values if p_value_col in chi_square_results.columns else None
    delta_values = [delta_results.loc[f, delta_col] for f in common_features]
    
    # Color bars by significance if p-values available
    if p_values is not None:
        colors = ['indianred' if p < significance_level else 'lightcoral' for p in p_values]
    else:
        colors = 'indianred'
    
    fig = go.Figure()
    
    # Add bar chart for chi-squared statistics (left y-axis)
    fig.add_trace(go.Bar(
        x=common_features,
        y=chi2_values,
        name='Chi-Squared',
        text=[f"{x:.2f}" for x in chi2_values],
        textposition="inside",
        marker_color=colors,
        yaxis='y1'
    ))
    
    # Add line chart for deltas (right y-axis)
    fig.add_trace(go.Scatter(
        x=common_features,
        y=delta_values,
        name='Delta %',
        mode='lines+markers',
        marker=dict(color="darkorange", size=10),
        line=dict(color="darkorange", width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title=xaxis_title),
        yaxis=dict(
            title=y1_title,
            titlefont=dict(color='indianred'),
            tickfont=dict(color='indianred')
        ),
        yaxis2=dict(
            title=y2_title,
            titlefont=dict(color='darkorange'),
            tickfont=dict(color='darkorange'),
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.5,  # Center horizontally
            y=-0.15,  # Below the plot
            xanchor='center',  # Anchor point for x
            yanchor='top',  # Anchor point for y
            orientation='h',  # Horizontal layout
            bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent white background
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='x unified',
        margin=dict(b=80)  # Add bottom margin to accommodate legend
    )
    
    return fig


def create_dual_axis_statistical_chart(
    test_results_df: pd.DataFrame,
    count_diff_col: str = "count_difference_nrm",
    t_stat_col: str = "t_stat_welch",
    title: str = "Ave Wonky Count Difference & Welch's t-statistic by Feature",
    xaxis_title: str = "Feature",
    y1_title: str = "Count Difference (Wonky - Non-wonky)",
    y2_title: str = "Welch's t-statistic"
) -> go.Figure:
    """
    Create dual-axis chart: bars for count differences, line for Welch's t-statistic.
    
    Parameters:
    -----------
    test_results_df : pd.DataFrame
        DataFrame with features as index and count_diff/t_stat columns
    count_diff_col : str
        Column name for count difference (normalized)
    t_stat_col : str
        Column name for Welch's t-statistic
    title : str
        Chart title
    xaxis_title : str
        X-axis title
    y1_title : str
        Left Y-axis title (for bars)
    y2_title : str
        Right Y-axis title (for line)
        
    Returns:
    --------
    go.Figure
        Plotly figure with dual-axis chart
    """
    # Prepare data
    if test_results_df.index.name is None:
        features = test_results_df.index.tolist()
    else:
        features = test_results_df.index.tolist()
    
    count_diff = test_results_df[count_diff_col].values
    t_stat = test_results_df[t_stat_col].values
    
    fig = go.Figure()
    
    # Add bar chart for count differences (left y-axis)
    fig.add_trace(go.Bar(
        x=features,
        y=count_diff,
        name='Count Difference',
        marker_color='steelblue',
        yaxis='y1'
    ))
    
    # Add line chart for Welch's t-statistic (right y-axis)
    fig.add_trace(go.Scatter(
        x=features,
        y=t_stat,
        name="Welch's t-statistic",
        mode='lines+markers',
        marker=dict(color="darkorange", size=10),
        line=dict(color="darkorange", width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title=xaxis_title),
        yaxis=dict(
            title=y1_title,
            titlefont=dict(color='steelblue'),
            tickfont=dict(color='steelblue')
        ),
        yaxis2=dict(
            title=y2_title,
            titlefont=dict(color='darkorange'),
            tickfont=dict(color='darkorange'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    return fig


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
        # Use numeric columns by default
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


def create_distribution_comparison(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    group1_value: int = 1,
    group2_value: int = 0,
    plot_type: str = "histogram",
    group1_name: str = "Wonky",
    group2_name: str = "Non-Wonky",
    title: Optional[str] = None,
    nbins: int = 50,
    opacity: float = 0.7
) -> go.Figure:
    """
    Create histogram or box plot comparing wonky vs non-wonky distributions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with feature and group columns
    feature : str
        Column name for the feature to plot
    group_col : str
        Column name for group labels
    group1_value : int
        Value for first group
    group2_value : int
        Value for second group
    plot_type : str
        Type of plot: "histogram" or "box"
    group1_name : str
        Display name for first group
    group2_name : str
        Display name for second group
    title : str, optional
        Plot title
    nbins : int
        Number of bins for histogram
    opacity : float
        Opacity for histogram overlay
        
    Returns:
    --------
    go.Figure
        Plotly figure with distribution comparison
    """
    # Check which columns are missing and provide helpful error messages
    missing_cols = []
    if feature not in df.columns:
        missing_cols.append(f"feature '{feature}'")
        # Try to find similar column names
        similar_features = [col for col in df.columns 
                           if feature.split('_')[0].lower() in col.lower() 
                           or any(word in col.lower() for word in feature.split('_') if len(word) > 3)]
        if similar_features:
            raise ValueError(
                f"Feature '{feature}' not found in DataFrame. "
                f"Available similar columns: {similar_features[:10]}. "
                f"All columns: {list(df.columns)[:30]}"
            )
    if group_col not in df.columns:
        missing_cols.append(f"group column '{group_col}'")
        # Try to find similar column names
        similar_group_cols = [col for col in df.columns 
                            if 'wonky' in col.lower() or 'group' in col.lower()]
        if similar_group_cols:
            raise ValueError(
                f"Group column '{group_col}' not found in DataFrame. "
                f"Available similar columns: {similar_group_cols[:10]}. "
                f"All columns: {list(df.columns)[:30]}"
            )
    
    if missing_cols:
        missing_str = " and ".join(missing_cols)
        raise ValueError(
            f"{missing_str.capitalize()} not found in DataFrame. "
            f"Available columns: {list(df.columns)[:30]}"
        )
    
    group1_data = df[df[group_col] == group1_value][feature].dropna()
    group2_data = df[df[group_col] == group2_value][feature].dropna()
    
    if plot_type == "histogram":
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=group1_data,
            name=group1_name,
            opacity=opacity,
            nbinsx=nbins,
            marker_color='crimson'
        ))
        
        fig.add_trace(go.Histogram(
            x=group2_data,
            name=group2_name,
            opacity=opacity,
            nbinsx=nbins,
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            barmode='overlay',
            title=title or f"{feature} Distribution: {group1_name} vs {group2_name}",
            xaxis_title=feature,
            yaxis_title="Frequency"
        )
        
    elif plot_type == "box":
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=group1_data,
            name=group1_name,
            marker_color='crimson',
            boxmean='sd'
        ))
        
        fig.add_trace(go.Box(
            y=group2_data,
            name=group2_name,
            marker_color='steelblue',
            boxmean='sd'
        ))
        
        fig.update_layout(
            title=title or f"{feature} Distribution: {group1_name} vs {group2_name}",
            yaxis_title=feature,
            xaxis_title="Group"
        )
    
    else:
        raise ValueError(f"plot_type must be 'histogram' or 'box', got '{plot_type}'")
    
    return fig

