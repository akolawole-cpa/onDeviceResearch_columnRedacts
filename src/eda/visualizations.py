"""
Visualizations Module

Helper functions for creating plotly visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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

