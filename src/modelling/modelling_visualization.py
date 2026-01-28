"""
Model Visualization - Plotly charts for feature importance analysis

Charts for comparing SHAP importance, OLS coefficients, contributions, and interactions.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def plot_feature_comparison(
    summary_df: pd.DataFrame,
    top_n: int = 15,
    data_cols: list = ["shap_importance", "ols_coefficient"],
    label_cols: list = ["shap_importance", "ols_coefficient"],
) -> go.Figure:
    """
    Dual bar chart comparing SHAP importance and OLS coefficients.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Feature summary with 'feature', 'shap_importance', 'ols_coefficient'
    top_n : int
        Number of top features to display
    data_cols : List
        List two columns from the summary that you'd like charted
    label_cols : List
        List two columns from the summary that you'd like charted
    
    Returns:
    --------
    go.Figure - Plotly figure
    """

    # Top N by SHAP importance
    plot_df = summary_df.head(top_n).copy()
    plot_df = plot_df.sort_values('shap_importance', ascending=True)
    
    fig = go.Figure()
    
    

    # 1st item
    fig.add_trace(go.Bar(
        y=plot_df['feature'],
        x=plot_df[data_cols[0]],
        name=label_cols[0],
        orientation='h',
        text=plot_df[label_cols[0]],
        textposition='outside',
        textfont=dict(size=10),
    ))
    
    # 2nd item
    fig.add_trace(go.Bar(
        y=plot_df['feature'],
        x=plot_df[data_cols[1]],
        name=label_cols[1],
        orientation='h',
        text=plot_df[label_cols[1]],
        textposition='outside',
        textfont=dict(size=10),
    ))
    
    fig.update_layout(
        title=dict(text=f"Snap comparisons - Top {top_n}", font=dict(size=16)),
        barmode='group',
        bargap=0.10,
        bargroupgap=0.075,
        xaxis_title="Value",
        yaxis_title="Feature",
        height=max(600, top_n * 40),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        template='plotly_dark',
        margin=dict(l=150, r=80),
    )
    
    fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=1)
    
    return fig


def plot_shap_contributions(
    summary_df: pd.DataFrame, 
    top_n: int = 20,
) -> go.Figure:
    """
    Horizontal bar chart of SHAP contributions showing direction of effect.
    
    Red = increases wonkiness, Green = decreases wonkiness
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Feature summary with 'feature', 'mean_contribution'
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    go.Figure - Plotly figure
    """
    plot_df = summary_df.head(top_n).copy()
    plot_df = plot_df.sort_values('mean_contribution', ascending=True)
    
    colors = ['#E74C3C' if x > 0 else '#27AE60' for x in plot_df['mean_contribution']]
    
    fig = go.Figure(go.Bar(
        y=plot_df['feature'],
        x=plot_df['mean_contribution'],
        orientation='h',
        marker_color=colors,
        text=plot_df['mean_contribution'].round(4),
        textposition='outside',
    ))
    
    fig.update_layout(
        title="SHAP Contributions to Wonkiness",
        xaxis_title="Mean SHAP Contribution",
        yaxis_title="Feature",
        height=max(400, top_n * 35),
        template='plotly_dark',
        margin=dict(l=150, r=80),
        annotations=[
            dict(x=0.02, y=1.05, xref='paper', yref='paper',
                 text='🔴 Increases wonkiness | 🟢 Decreases wonkiness',
                 showarrow=False, font=dict(size=11))
        ]
    )
    
    fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=1)
    
    return fig


def plot_interactions(
    interaction_df: pd.DataFrame, 
    top_n: int = 15,
) -> go.Figure:
    """
    Bar chart of top feature interactions.
    
    Parameters:
    -----------
    interaction_df : pd.DataFrame
        Interaction summary with 'feature_1', 'feature_2', 'interaction_strength'
    top_n : int
        Number of top interactions to display
    
    Returns:
    --------
    go.Figure - Plotly figure
    """
    plot_df = interaction_df.head(top_n).copy()
    plot_df['pair'] = plot_df['feature_1'] + ' × ' + plot_df['feature_2']
    plot_df = plot_df.sort_values('interaction_strength', ascending=True)
    
    fig = go.Figure(go.Bar(
        y=plot_df['pair'],
        x=plot_df['interaction_strength'],
        orientation='h',
        marker_color='#9B59B6',
        text=plot_df['interaction_strength'].round(4),
        textposition='outside',
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Interactions",
        xaxis_title="Interaction Strength (Mean |SHAP Interaction|)",
        yaxis_title="Feature Pair",
        height=max(400, top_n * 35),
        template='plotly_dark',
        margin=dict(l=250, r=80),
    )
    
    return fig