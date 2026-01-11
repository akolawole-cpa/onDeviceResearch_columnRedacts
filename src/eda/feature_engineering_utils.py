"""utils for feature engineering"""


import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict


def one_hot_encode_column(
    df: pd.DataFrame,
    column: str,
    prefix: Optional[str] = None,
    as_string: bool = False,
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    One-hot encode a column (including multi-value/list columns) and join to dataframe.
    
    Replaces this repeated pattern:
        series = df['column']
        dummies = (
            series.explode()
             .str.strip()  # or .astype(str).str.strip()
             .pipe(pd.get_dummies)
             .groupby(level=0).sum()
        )
        dummies = dummies.add_prefix('column_')
        df = df.join(dummies)
    
    Args:
        df: Input DataFrame
        column: Column name to encode
        prefix: Prefix for dummy columns (defaults to column name + '_')
        as_string: If True, convert values to string first (useful for numeric columns)
    
    Returns:
        Tuple of (DataFrame with dummies joined, Index of new column names)
    
    Examples:
        # Basic usage
        df, cols = one_hot_encode_column(df, 'gambling_participation_mc')
        
        # With custom prefix
        df, cols = one_hot_encode_column(df, 'email_verified', prefix='email_veri_')
        
        # For numeric columns
        df, cols = one_hot_encode_column(df, 'quality', as_string=True)
    """
    if prefix is None:
        prefix = f"{column}_"
    
    series = df[column]
    
    if as_string:
        dummies = (
            series.explode()
            .astype(str).str.strip()
            .pipe(pd.get_dummies)
            .groupby(level=0).sum()
        )
    else:
        dummies = (
            series.explode()
            .str.strip()
            .pipe(pd.get_dummies)
            .groupby(level=0).sum()
        )
    
    dummies = dummies.add_prefix(prefix)
    new_cols = dummies.columns
    
    df = df.join(dummies)
    
    return df, new_cols


def create_threshold_features(
    df: pd.DataFrame,
    column: str,
    thresholds: List[Tuple[str, str, float]],
) -> pd.DataFrame:
    """
    Create binary features based on threshold comparisons.
    
    Replaces this repeated pattern:
        df["quality=100"] = np.where(df["quality"] == 100, 1, 0)
        df["quality<90"] = np.where(df["quality"] < 90, 1, 0)
        df["quality<75"] = np.where(df["quality"] < 75, 1, 0)
        df["risk<50"] = np.where(df["risk"] < 50, 1, 0)
    
    Args:
        df: Input DataFrame
        column: Column name to create thresholds from
        thresholds: List of tuples (new_column_name, operator, value)
                   operator can be: '<', '<=', '>', '>=', '==', '!='
    
    Returns:
        DataFrame with new threshold columns added
    
    Example:
        thresholds = [
            ('quality=100', '==', 100),
            ('quality<90', '<', 90),
            ('quality<75', '<', 75),
            ('quality<50', '<', 50),
            ('quality<30', '<', 30),
        ]
        df = create_threshold_features(df, 'quality', thresholds)
    """
    ops = {
        '<': lambda s, v: s < v,
        '<=': lambda s, v: s <= v,
        '>': lambda s, v: s > v,
        '>=': lambda s, v: s >= v,
        '==': lambda s, v: s == v,
        '!=': lambda s, v: s != v,
    }
    
    new_features = {}
    for new_col, op, value in thresholds:
        if op not in ops:
            raise ValueError(f"Unknown operator: {op}. Use one of {list(ops.keys())}")
        new_features[new_col] = np.where(ops[op](df[column], value), 1, 0)
    
    # Use pd.concat to avoid fragmentation warnings
    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)


def create_delta_category_features(
    df: pd.DataFrame,
    delta_column: str,
    prefix: Optional[str] = None,
    large_threshold: float = 50,
) -> pd.DataFrame:
    """
    Create categorical features from a delta/change column.
    
    Replaces this repeated pattern:
        df['quality_delta_LargePostive'] = np.where(df['quality_delta'] > 50, 1, 0)
        df['quality_delta_Postive'] = np.where(df['quality_delta'] > 0, 1, 0)
        df['quality_delta_Neutral'] = np.where(df['quality_delta'] == 0, 1, 0)
        df['quality_delta_LargeNegative'] = np.where(df['quality_delta'] < -50, 1, 0)
        df['quality_delta_Negative'] = np.where(df['quality_delta'] < 0, 1, 0)
    
    Args:
        df: Input DataFrame
        delta_column: Column containing the delta values
        prefix: Prefix for new columns (defaults to delta_column + '_')
        large_threshold: Threshold for "large" positive/negative (default 50)
    
    Returns:
        DataFrame with new category columns added
    
    Example:
        df = create_delta_category_features(df, 'quality_delta')
        df = create_delta_category_features(df, 'risk_delta', large_threshold=30)
    """
    if prefix is None:
        prefix = f"{delta_column}_"
    
    new_features = {
        f'{prefix}LargePositive': np.where(df[delta_column] > large_threshold, 1, 0),
        f'{prefix}Positive': np.where(df[delta_column] > 0, 1, 0),
        f'{prefix}Neutral': np.where(df[delta_column] == 0, 1, 0),
        f'{prefix}LargeNegative': np.where(df[delta_column] < -large_threshold, 1, 0),
        f'{prefix}Negative': np.where(df[delta_column] < 0, 1, 0),
    }
    
    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)


def create_binned_features(
    df: pd.DataFrame,
    column: str,
    bins: List[int],
    prefix: Optional[str] = None,
    include_ge_last: bool = True,
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Create binned features with cumulative "<X" columns and a ">=X" column.
    
    Replaces this repeated pattern:
        bins = [2, 7, 14, 21, 28, 31, 50]
        for limit in bins:
            dummies[f'<{limit}'] = dummies[[col for col in dummies.columns 
                if col.isdigit() and int(col) < limit]].sum(axis=1)
        dummies[f'>=_50'] = dummies[[col for col in dummies.columns 
            if col.isdigit() and int(col) >= 50]].sum(axis=1)
    
    Args:
        df: Input DataFrame
        column: Column name to bin
        bins: List of bin thresholds (e.g., [2, 7, 14, 21, 28, 31, 50])
        prefix: Prefix for new columns (defaults to column name + '_')
        include_ge_last: If True, include a '>= last_bin' column
    
    Returns:
        Tuple of (DataFrame with binned columns, Index of new column names)
    
    Example:
        df, cols = create_binned_features(
            df, 'days_active_before_task', 
            bins=[2, 7, 14, 21, 28, 31, 50],
            prefix='days_active_'
        )
    """
    if prefix is None:
        prefix = f"{column}_"
    
    new_features = {}
    
    for limit in bins:
        new_features[f'{prefix}<{limit}'] = np.where(df[column] < limit, 1, 0)
    
    if include_ge_last and bins:
        last_bin = bins[-1]
        new_features[f'{prefix}>={last_bin}'] = np.where(df[column] >= last_bin, 1, 0)
    
    new_df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    new_cols = pd.Index(new_features.keys())
    
    return new_df, new_cols


def batch_one_hot_encode(
    df: pd.DataFrame,
    columns: List[str],
    as_string_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    One-hot encode multiple columns at once.
    
    More efficient than encoding one at a time as it avoids DataFrame fragmentation.
    
    Args:
        df: Input DataFrame
        columns: List of column names to encode
        as_string_columns: Columns that should be converted to string first
    
    Returns:
        DataFrame with all dummy columns added
    
    Example:
        df = batch_one_hot_encode(
            df,
            columns=['email_verified', 'notify_task_payout', 'notify_new_task', 
                    'share_location_data', 'exposure_band', 'gender'],
            as_string_columns=['quality', 'risk']
        )
    """
    if as_string_columns is None:
        as_string_columns = []
    
    all_dummies = []
    
    for col in columns:
        as_str = col in as_string_columns
        series = df[col]
        
        if as_str:
            dummies = (
                series.explode()
                .astype(str).str.strip()
                .pipe(pd.get_dummies)
                .groupby(level=0).sum()
            )
        else:
            dummies = (
                series.explode()
                .str.strip()
                .pipe(pd.get_dummies)
                .groupby(level=0).sum()
            )
        
        dummies = dummies.add_prefix(f'{col}_')
        all_dummies.append(dummies)
    
    if all_dummies:
        combined_dummies = pd.concat(all_dummies, axis=1)
        df = pd.concat([df, combined_dummies], axis=1)
    
    return df


def create_value_mapping_and_encode(
    df: pd.DataFrame,
    column: str,
    mapping: Dict,
    new_column: Optional[str] = None,
    one_hot: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Index]]:
    """
    Map values and optionally one-hot encode the result.
    
    Replaces this repeated pattern:
        income_map = {'Less than £15,000': 'Less than £15,000', ...}
        df["fulcrum_household_income_mapped"] = df["fulcrum_household_income"].map(income_map)
        dummies = (series.explode()...)
        df = df.join(dummies)
    
    Args:
        df: Input DataFrame
        column: Column name to map
        mapping: Dictionary mapping old values to new values
        new_column: Name for mapped column (defaults to column + '_mapped')
        one_hot: If True, also one-hot encode the mapped column
    
    Returns:
        Tuple of (DataFrame with mapped/encoded columns, Index of dummy columns or None)
    
    Example:
        income_map = {
            'Less than £15,000': 'Less than £15,000',
            '£15,000 to £19,999': '£15,000 to £19,999',
            # ... etc
        }
        df, cols = create_value_mapping_and_encode(df, 'fulcrum_household_income', income_map)
    """
    if new_column is None:
        new_column = f"{column}_mapped"
    
    df[new_column] = df[column].map(mapping)
    
    if one_hot:
        df, cols = one_hot_encode_column(df, new_column, as_string=True)
        return df, cols
    
    return df, None


def create_score_features(
    df: pd.DataFrame,
    column: str,
    perfect_value: float = 100,
    thresholds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Create standard threshold features for score columns (quality, risk, etc.)
    
    Args:
        df: Input DataFrame
        column: Score column name
        perfect_value: Value considered "perfect" (default 100)
        thresholds: List of threshold values (default [90, 75, 50, 30])
    
    Returns:
        DataFrame with threshold features added
    
    Example:
        df = create_score_features(df, 'quality')
        df = create_score_features(df, 'risk', thresholds=[90, 80, 50])
    """
    if thresholds is None:
        thresholds = [90, 75, 50, 30]
    
    threshold_specs = [(f'{column}={int(perfect_value)}', '==', perfect_value)]
    threshold_specs.extend([
        (f'{column}<{t}', '<', t) for t in thresholds
    ])
    
    return create_threshold_features(df, column, threshold_specs)
  

def filter_to_engineered_features(
    df, 
    original_columns: set,
    id_column: str = 'respondentPk',
    additional_columns: list = None
):
    """
    Filter dataframe to keep only the ID column and engineered features.
    
    Args:
        df: DataFrame after feature engineering
        original_columns: Set of column names from before feature engineering
        id_column: Name of the ID column to keep (default: 'respondentPk')
        additional_columns: Optional list of additional columns to keep
    
    Returns:
        Filtered DataFrame with only ID and engineered columns
    
    Example:
        # At the start of notebook:
        original_columns = set(user_info_df.columns)
        
        # ... do all feature engineering ...
        
        # At the end:
        user_info_df_final = filter_to_engineered_features(
            user_info_df, 
            original_columns,
            id_column='respondentPk'
        )
    """
    engineered_cols = [col for col in df.columns if col not in original_columns]
    
    final_cols = [id_column] + engineered_cols
    
    if additional_columns:
        for col in additional_columns:
            if col in df.columns and col not in final_cols:
                final_cols.append(col)
    
    seen = set()
    final_cols = [x for x in final_cols if not (x in seen or seen.add(x))]
    
    return df[final_cols]