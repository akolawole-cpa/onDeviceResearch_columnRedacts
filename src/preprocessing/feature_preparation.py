"""
Feature Preparation Module

Feature preparation pipelines for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from typing import List, Optional, Dict, Tuple


def scale_features(
    X: np.ndarray,
    method: str = "standard",
    fit: bool = True,
    scaler: Optional = None
) -> Tuple[np.ndarray, object]:
    """
    Scale features using specified method.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    method : str
        Scaling method ("standard", "minmax", "robust")
    fit : bool
        Whether to fit the scaler
    scaler : object, optional
        Pre-fitted scaler
        
    Returns:
    --------
    tuple
        (scaled_X, scaler)
    """
    if scaler is None:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "label"
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Encode categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with categorical columns
    columns : List[str]
        List of categorical column names
    method : str
        Encoding method ("label" or "onehot")
        
    Returns:
    --------
    tuple
        (encoded_df, encoders_dict)
    """
    encoded_df = df.copy()
    encoders = {}
    
    if method == "label":
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    elif method == "onehot":
        encoded_df = pd.get_dummies(encoded_df, columns=columns, prefix=columns)
        encoders = {col: "onehot" for col in columns}
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return encoded_df, encoders


def prepare_features_for_modeling(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
    scaling_method: str = "standard",
    encoding_method: str = "label",
    remove_low_variance: bool = True,
    variance_threshold: float = 0.01
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Prepare features for modeling with scaling and encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : List[str]
        List of feature column names
    target_col : str, optional
        Name of target column
    categorical_cols : List[str], optional
        List of categorical column names
    scaling_method : str
        Scaling method
    encoding_method : str
        Encoding method for categoricals
    remove_low_variance : bool
        Whether to remove low variance features
    variance_threshold : float
        Variance threshold for removal
        
    Returns:
    --------
    tuple
        (X, y, preprocessing_info)
    """
    # Select features
    available_features = [f for f in feature_cols if f in df.columns]
    prep_df = df[available_features].copy()
    
    # Encode categoricals
    if categorical_cols:
        cat_cols = [c for c in categorical_cols if c in prep_df.columns]
        if cat_cols:
            prep_df, encoders = encode_categorical(prep_df, cat_cols, encoding_method)
    else:
        encoders = {}
    
    # Convert to numeric
    for col in prep_df.columns:
        prep_df[col] = pd.to_numeric(prep_df[col], errors='coerce')
    
    # Remove low variance
    if remove_low_variance:
        variances = prep_df.var()
        high_variance_cols = variances[variances >= variance_threshold].index.tolist()
        prep_df = prep_df[high_variance_cols]
    
    # Scale features
    X = prep_df.values
    X_scaled, scaler = scale_features(X, method=scaling_method)
    
    # Get target
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].values
    
    preprocessing_info = {
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': prep_df.columns.tolist(),
        'variance_threshold': variance_threshold
    }
    
    return X_scaled, y, preprocessing_info

