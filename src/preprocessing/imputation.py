"""
Imputation Module

Cluster-based null imputation functions.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from src.models.clustering import fit_clustering_model


def impute_using_cluster_mode(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    column: str
) -> pd.Series:
    """
    Impute null values using the mode of the cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with null values
    cluster_labels : np.ndarray
        Cluster assignments for each row
    column : str
        Column name to impute
        
    Returns:
    --------
    pd.Series
        Series with imputed values
    """
    imputed = df[column].copy()
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = df.loc[cluster_mask, column]
        
        # Get mode of non-null values in cluster
        non_null_values = cluster_data.dropna()
        if len(non_null_values) > 0:
            mode_value = non_null_values.mode()
            if len(mode_value) > 0:
                imputed.loc[cluster_mask & imputed.isna()] = mode_value.iloc[0]
    
    return imputed


def impute_using_cluster_mean(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    column: str
) -> pd.Series:
    """
    Impute null values using the mean of the cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with null values
    cluster_labels : np.ndarray
        Cluster assignments for each row
    column : str
        Column name to impute
        
    Returns:
    --------
    pd.Series
        Series with imputed values
    """
    imputed = df[column].copy()
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = df.loc[cluster_mask, column]
        
        # Get mean of non-null values in cluster
        mean_value = cluster_data.mean()
        if not np.isnan(mean_value):
            imputed.loc[cluster_mask & imputed.isna()] = mean_value
    
    return imputed


def impute_using_cluster_median(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    column: str
) -> pd.Series:
    """
    Impute null values using the median of the cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with null values
    cluster_labels : np.ndarray
        Cluster assignments for each row
    column : str
        Column name to impute
        
    Returns:
    --------
    pd.Series
        Series with imputed values
    """
    imputed = df[column].copy()
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = df.loc[cluster_mask, column]
        
        # Get median of non-null values in cluster
        median_value = cluster_data.median()
        if not np.isnan(median_value):
            imputed.loc[cluster_mask & imputed.isna()] = median_value
    
    return imputed


def cluster_based_imputation(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    clustering_features: List[str],
    clustering_method: str = "kmeans",
    impute_strategy: str = "cluster_mode",
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Perform cluster-based imputation for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with null values
    columns_to_impute : List[str]
        List of column names to impute
    clustering_features : List[str]
        List of feature columns to use for clustering
    clustering_method : str
        Clustering method ("kmeans", "dbscan", "hierarchical")
    impute_strategy : str
        Imputation strategy ("cluster_mode", "cluster_mean", "cluster_median")
    config : dict, optional
        Configuration dictionary for clustering
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with imputed values
    """
    # Prepare features for clustering
    available_features = [f for f in clustering_features if f in df.columns]
    
    if len(available_features) == 0:
        raise ValueError("No valid clustering features found")
    
    # Remove rows with nulls in clustering features for training
    clustering_data = df[available_features].dropna()
    
    if len(clustering_data) == 0:
        raise ValueError("No valid data for clustering")
    
    # Fit clustering model
    from src.models.clustering import fit_clustering_model
    
    model, cluster_labels_train = fit_clustering_model(
        clustering_data.values,
        method=clustering_method,
        config=config
    )
    
    # Predict clusters for all data (including rows with nulls in target columns)
    all_clustering_data = df[available_features].fillna(df[available_features].median())
    all_cluster_labels = model.predict(all_clustering_data.values)
    
    # Impute each column
    imputed_df = df.copy()
    
    for column in columns_to_impute:
        if column not in df.columns:
            continue
        
        if impute_strategy == "cluster_mode":
            imputed_df[column] = impute_using_cluster_mode(
                df,
                all_cluster_labels,
                column
            )
        elif impute_strategy == "cluster_mean":
            imputed_df[column] = impute_using_cluster_mean(
                df,
                all_cluster_labels,
                column
            )
        elif impute_strategy == "cluster_median":
            imputed_df[column] = impute_using_cluster_median(
                df,
                all_cluster_labels,
                column
            )
        else:
            raise ValueError(f"Unknown impute_strategy: {impute_strategy}")
    
    return imputed_df


def simple_imputation(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    strategy: str = "mean"
) -> pd.DataFrame:
    """
    Perform simple imputation (mean, median, mode) without clustering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with null values
    columns_to_impute : List[str]
        List of column names to impute
    strategy : str
        Imputation strategy ("mean", "median", "mode")
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with imputed values
    """
    imputed_df = df.copy()
    
    for column in columns_to_impute:
        if column not in df.columns:
            continue
        
        if strategy == "mean":
            imputed_df[column] = df[column].fillna(df[column].mean())
        elif strategy == "median":
            imputed_df[column] = df[column].fillna(df[column].median())
        elif strategy == "mode":
            mode_value = df[column].mode()
            if len(mode_value) > 0:
                imputed_df[column] = df[column].fillna(mode_value.iloc[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return imputed_df

