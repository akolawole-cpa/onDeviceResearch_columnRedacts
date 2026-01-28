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