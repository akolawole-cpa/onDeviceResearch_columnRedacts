"""utils for feature engineering"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Optional, Dict


def one_hot_encode_column(
    df: pd.DataFrame,
    column: str,
    prefix: Optional[str] = None,
    as_string: bool = False,
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    One-hot encode a column (including multi-value/list columns)
     and join to dataframe.

    Args:
        df: Input DataFrame
        column: Column name to encode
        prefix: Prefix for dummy columns
         (defaults to column name + '_')
        as_string: If True, convert values to string first
         (useful for numeric columns).

    Returns:
        Tuple of (DataFrame with dummies joined, Index of new column names)

    Examples:
        df, cols = one_hot_encode_column(df, 'gambling_participation_mc')
    """
    if prefix is None:
        prefix = f"{column}_"

    series = df[column]

    if as_string:
        dummies = (
            series.explode()
            .astype(str)
            .str.strip()
            .pipe(pd.get_dummies)
            .groupby(level=0)
            .sum()
        )
    else:
        dummies = (
            series.explode().str.strip().pipe(pd.get_dummies).groupby(level=0).sum()
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
        "<": lambda s, v: s < v,
        "<=": lambda s, v: s <= v,
        ">": lambda s, v: s > v,
        ">=": lambda s, v: s >= v,
        "==": lambda s, v: s == v,
        "!=": lambda s, v: s != v,
    }

    new_features = {}
    for new_col, op, value in thresholds:
        if op not in ops:
            raise ValueError(f"Unknown operator: {op}. Use one of {list(ops.keys())}")
        new_features[new_col] = np.where(ops[op](df[column], value), 1, 0)

    # Use pd.concat to avoid fragmentation warnings
    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)


def create_binned_features(
    df: pd.DataFrame,
    column: str,
    bins: List[int],
    prefix: Optional[str] = None,
    include_ge_last: bool = True,
    include_mutually_exclusive: bool = False,
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Create binned features with cumulative "<X" columns and a ">=X" column.
    Optionally also creates mutually exclusive "between" bins.

    Args:
        df: Input DataFrame
        column: Column name to bin
        bins: List of bin thresholds (e.g., [2, 7, 14, 21, 28, 31, 50])
        prefix: Prefix for new columns (defaults to column name + '_')
        include_ge_last: If True, include a '>= last_bin' column
        include_mutually_exclusive: If True, also create non-overlapping bins
            (e.g., '0_to_2', '3_to_10', '11_to_20', etc.)

    Returns:
        Tuple of (DataFrame with binned columns, Index of new column names)

    Example:
        # Cumulative bins only (default)
        df, cols = create_binned_features(
            df, 'days_active_before_task',
            bins=[2, 10, 20, 30, 50],
            prefix='days_active_'
        )
        # Creates: days_active_<2, days_active_<10, ..., days_active_>=50

        # With mutually exclusive bins
        df, cols = create_binned_features(
            df, 'days_active_before_task',
            bins=[2, 10, 20, 30, 50],
            prefix='days_active_',
            include_mutually_exclusive=True
        )
        # Also creates: days_active_0_to_2, days_active_3_to_10, ..., days_active_51_plus
    """
    if prefix is None:
        prefix = f"{column}_"

    new_features = {}

    for limit in bins:
        new_features[f"{prefix}<{limit}"] = np.where(df[column] < limit, 1, 0)

    if include_ge_last and bins:
        last_bin = bins[-1]
        new_features[f"{prefix}>={last_bin}"] = np.where(df[column] >= last_bin, 1, 0)

    if include_mutually_exclusive:
        boundaries = [0] + bins

        for i in range(len(boundaries)):
            lower = boundaries[i]

            if i < len(boundaries) - 1:
                upper = boundaries[i + 1]

                if i == 0:
                    col_name = f"{prefix}{lower}_to_{upper}"
                    new_features[col_name] = np.where(
                        (df[column] >= lower) & (df[column] <= upper), 1, 0
                    )
                else:
                    actual_lower = lower + 1
                    col_name = f"{prefix}{actual_lower}_to_{upper}"
                    new_features[col_name] = np.where(
                        (df[column] >= actual_lower) & (df[column] <= upper), 1, 0
                    )
            else:
                actual_lower = lower + 1
                col_name = f"{prefix}{actual_lower}_plus"
                new_features[col_name] = np.where(df[column] >= actual_lower, 1, 0)

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
                .astype(str)
                .str.strip()
                .pipe(pd.get_dummies)
                .groupby(level=0)
                .sum()
            )
        else:
            dummies = (
                series.explode().str.strip().pipe(pd.get_dummies).groupby(level=0).sum()
            )

        dummies = dummies.add_prefix(f"{col}_")
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

    threshold_specs = [(f"{column}={int(perfect_value)}", "==", perfect_value)]
    threshold_specs.extend([(f"{column}<{t}", "<", t) for t in thresholds])

    return create_threshold_features(df, column, threshold_specs)


def filter_to_engineered_features(
    df,
    original_columns: set,
    id_column: str = "respondentPk",
    kpi_column: str = "wonky_study_count",
    additional_columns: list = None,
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
            id_column='respondentPk',
            kpi_column: str = 'wonky_study_count'
        )
    """
    engineered_cols = [col for col in df.columns if col not in original_columns]

    final_cols = [id_column] + [kpi_column] + engineered_cols

    if additional_columns:
        for col in additional_columns:
            if col in df.columns and col not in final_cols:
                final_cols.append(col)

    seen = set()
    final_cols = [x for x in final_cols if not (x in seen or seen.add(x))]

    return df[final_cols]


def map_hardware_to_category(hardware: Optional[str]) -> str:
    """
    Map a hardware SKU string to a simplified category.

    Reduces 200+ unique hardware SKUs to ~15 meaningful categories based on:
    - iPhone model generations
    - Chipset manufacturer and tier (MediaTek, Exynos, Qualcomm)
    - Google Pixel codenames
    - Other identifiable patterns

    Parameters
    ----------
    hardware : str or None
        Raw hardware SKU string (e.g., 'iPhone14,2', 'mt6768', 'shiba')

    Returns
    -------
    str
        Simplified category (e.g., 'iphone_14', 'mediatek_mid', 'google_pixel')
    """
    if hardware is None or pd.isna(hardware):
        return "unknown"

    hw = str(hardware).lower().strip()

    # === APPLE DEVICES ===
    if "iphone" in hw:
        match = re.search(r"iphone\s*(\d+)", hw)
        if match:
            gen = int(match.group(1))
            if gen >= 17:
                return "iphone_16"
            elif gen >= 15:
                return "iphone_15"
            elif gen >= 14:
                return "iphone_14"
            elif gen >= 13:
                return "iphone_13"
            else:
                return "iphone_legacy"

        if "se" in hw:
            return "iphone_se"
        elif "pro max" in hw or "11 pro max" in hw:
            return "iphone_pro_max"
        elif "pro" in hw:
            return "iphone_pro"
        elif any(x in hw for x in ["xs max", "xr", "xs", "x"]):
            return "iphone_x_series"
        elif any(x in hw for x in ["11", "12"]):
            return "iphone_11_12"
        elif any(x in hw for x in ["6", "7", "8"]):
            return "iphone_legacy"
        else:
            return "iphone_other"

    # === MEDIATEK CHIPSETS ===
    if hw.startswith("mt") or "mediatek" in hw:
        match = re.search(r"mt(\d+)", hw)
        if match:
            model = int(match.group(1))
            if model >= 6980:
                return "mediatek_flagship"
            elif model >= 6850 or model in range(6890, 6900):
                return "mediatek_upper_mid"
            elif model >= 6780:
                return "mediatek_mid"
            else:
                return "mediatek_budget"
        return "mediatek_other"

    # === SAMSUNG EXYNOS ===
    if "exynos" in hw or hw.startswith("s5e"):
        if any(x in hw for x in ["2100", "2200", "990", "9945", "9925", "9955"]):
            return "exynos_flagship"
        elif any(x in hw for x in ["980", "9825", "9820", "9810"]):
            return "exynos_upper_mid"
        elif any(x in hw for x in ["9611", "9610", "850", "8895", "8835", "8825"]):
            return "exynos_mid"
        else:
            return "exynos_budget"

    # === QUALCOMM ===
    if hw == "qcom" or "qualcomm" in hw or "snapdragon" in hw:
        return "qualcomm"

    # === GOOGLE PIXEL (codenames) ===
    pixel_codenames = {
        "caiman",
        "komodo",
        "tokay",
        "comet",  # Pixel 9
        "shiba",
        "husky",
        "akita",  # Pixel 8
        "panther",
        "cheetah",
        "lynx",  # Pixel 7
        "oriole",
        "raven",
        "bluejay",  # Pixel 6
        "redfin",
        "bramble",
        "sunfish",
        "flame",
        "coral",
        "bonito",
        "sargo",
        "crosshatch",
        "blueline",
        "taimen",
        "walleye",
        "felix",
        "tegu",
    }
    if hw in pixel_codenames:
        return "google_pixel"

    # === UNISOC (Spreadtrum) ===
    if hw.startswith("ums") or hw.startswith("sp") or "s9863a" in hw:
        return "unisoc_budget"

    # === HUAWEI KIRIN ===
    if "kirin" in hw or "hi62" in hw or "hi32" in hw or "hi36" in hw:
        match = re.search(r"kirin\s*(\d+)", hw)
        if match:
            model = int(match.group(1))
            if model >= 980:
                return "kirin_flagship"
            else:
                return "kirin_mid"
        return "kirin_other"

    # === OTHER KNOWN BRANDS ===
    if "samsung" in hw:
        return "samsung_other"
    if "xiaomi" in hw:
        return "xiaomi"
    if "rmx" in hw or "realme" in hw:
        return "realme"
    if any(x in hw for x in ["lg", "lucye", "joan", "judyln", "elsa", "winglm"]):
        return "lg"
    if "itel" in hw or hw.startswith("w6") or hw.startswith("w5"):
        return "itel_budget"
    if "tcl" in hw:
        return "tcl"
    if "surface" in hw:
        return "microsoft_surface"

    return "other"


def map_manufacturer_to_category(manufacturer: Optional[str]) -> str:
    """
    Map a manufacturer string to a simplified/normalized category.

    Consolidates variations (e.g., 'samsung', 'Samsung' → 'samsung')
    and groups smaller brands into meaningful categories.

    Parameters
    ----------
    manufacturer : str or None
        Raw manufacturer string

    Returns
    -------
    str
        Normalized manufacturer category
    """
    if manufacturer is None or pd.isna(manufacturer):
        return "unknown"

    mfr = str(manufacturer).lower().strip()

    # === MAJOR BRANDS ===
    if mfr in ["apple"]:
        return "apple"
    if mfr in ["samsung"]:
        return "samsung"
    if mfr in ["google"]:
        return "google"
    if mfr in ["xiaomi", "xiaomi communications co ltd", "xiaomi inc"]:
        return "xiaomi"
    if mfr in ["huawei", "honor"]:
        return "huawei_honor"
    if mfr in ["oppo"]:
        return "oppo"
    if mfr in ["vivo"]:
        return "vivo"
    if mfr in ["oneplus"]:
        return "oneplus"
    if mfr in ["motorola", "motorola mobility llc"]:
        return "motorola"
    if mfr in ["realme"]:
        return "realme"
    if mfr in ["nothing"]:
        return "nothing"
    if mfr in ["sony"]:
        return "sony"

    # === TRANSSION GROUP (Tecno, Infinix, Itel) ===
    if any(x in mfr for x in ["tecno", "infinix", "itel"]):
        return "transsion"

    # === BUDGET/CHINESE BRANDS ===
    budget_chinese = [
        "oscal",
        "cubot",
        "doogee",
        "oukitel",
        "blackview",
        "ulefone",
        "fossibot",
        "umidigi",
        "xgody",
        "gionee",
        "hxy",
        "villaon",
        "hena",
        "eacrugged",
        "agm",
        "firefly_mobile",
        "colors_mobile",
        "chinoe",
        "kxd",
        "gtel",
        "x-tigi",
        "qubo",
    ]
    if any(x in mfr for x in budget_chinese):
        return "budget_chinese"

    # === OTHER KNOWN BRANDS ===
    if mfr in ["tcl", "alcatel"]:
        return "tcl"
    if mfr in ["hmd global", "nokia"]:
        return "hmd_nokia"
    if mfr in ["lenovo"]:
        return "lenovo"
    if mfr in ["zte", "nubia"]:
        return "zte_nubia"
    if mfr in ["asus"]:
        return "asus"
    if mfr in ["lge", "lg"]:
        return "lg"
    if mfr in ["htc"]:
        return "htc"
    if mfr in ["meizu"]:
        return "meizu"
    if mfr in ["fairphone"]:
        return "fairphone"
    if mfr in ["microsoft"]:
        return "microsoft"
    if mfr in ["wiko"]:
        return "wiko"
    if mfr in ["hisense"]:
        return "hisense"

    # === RUGGED/ENTERPRISE ===
    if any(x in mfr for x in ["crosscall", "bullitt", "kyocera", "cat"]):
        return "rugged_enterprise"

    # === JAPANESE ===
    if mfr in ["fcnt", "sharp"]:
        return "japanese_other"

    # === TABLETS/OTHER ===
    if any(x in mfr for x in ["teclast", "pad", "tablet"]):
        return "tablet_other"

    # === ODM/REFERENCE (not real manufacturers) ===
    if mfr in ["sprd", "wheatek", "alps", "incar", "stack", "revoview", "fih", "img"]:
        return "odm_reference"

    return "other"


def add_hardware_category(
    df: pd.DataFrame,
    hardware_col: str = "ditr_hardware",
    new_col: str = "hardware_category",
) -> pd.DataFrame:
    """
    Add a hardware category column to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing hardware SKU column
    hardware_col : str
        Name of the column containing raw hardware SKUs
    new_col : str
        Name of the new category column to create

    Returns
    -------
    pd.DataFrame
        DataFrame with new category column added

    Example
    -------
    >>> user_info_df = add_hardware_category(user_info_df)
    >>> user_info_df['hardware_category'].value_counts()
    """
    df = df.copy()
    df[new_col] = df[hardware_col].apply(map_hardware_to_category)
    return df


def add_manufacturer_category(
    df: pd.DataFrame,
    manufacturer_col: str = "ditr_manufacturer",
    new_col: str = "manufacturer_category",
) -> pd.DataFrame:
    """
    Add a manufacturer category column to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing manufacturer column
    manufacturer_col : str
        Name of the column containing raw manufacturer names
    new_col : str
        Name of the new category column to create

    Returns
    -------
    pd.DataFrame
        DataFrame with new category column added

    Example
    -------
    >>> user_info_df = add_manufacturer_category(user_info_df)
    >>> user_info_df['manufacturer_category'].value_counts()
    """
    df = df.copy()
    df[new_col] = df[manufacturer_col].apply(map_manufacturer_to_category)
    return df


def add_device_categories(
    df: pd.DataFrame,
    hardware_col: str = "ditr_hardware",
    manufacturer_col: str = "ditr_manufacturer",
) -> pd.DataFrame:
    """
    Add both hardware and manufacturer category columns in one call.

    This is a convenience function that applies both mappings, reducing
    200+ hardware SKUs and 90+ manufacturers to ~15-20 categories each.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing device columns
    hardware_col : str
        Name of hardware SKU column
    manufacturer_col : str
        Name of manufacturer column

    Returns
    -------
    pd.DataFrame
        DataFrame with hardware_category and manufacturer_category columns added

    Example
    -------
    >>> user_info_df = add_device_categories(user_info_df)
    >>> print(user_info_df['hardware_category'].value_counts())
    >>> print(user_info_df['manufacturer_category'].value_counts())
    """
    df = add_hardware_category(df, hardware_col)
    df = add_manufacturer_category(df, manufacturer_col)
    return df


def create_recency_features(
    df: pd.DataFrame,
    last_seen_col: str,
    date_created_col: str,
    thresholds: List[int] = [2, 7],
    prefix: str = "recency_diff_",
    include_cumulative: bool = True,
    include_mutually_exclusive: bool = True,
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Create features based on the difference between last_seen and date_created.

    Computes the day difference (last_seen - date_created) and creates
    threshold-based binary features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    last_seen_col : str
        Column name for last seen date
    date_created_col : str
        Column name for date created
    thresholds : List[int]
        List of day thresholds (default: [2, 7])
    prefix : str
        Prefix for new columns (default: 'recency_diff_')
    include_cumulative : bool
        If True, create cumulative features (<=2, <=7)
    include_mutually_exclusive : bool
        If True, create mutually exclusive bins (0-2, 3-7, 8+)

    Returns
    -------
    Tuple[pd.DataFrame, pd.Index]
        (DataFrame with features, Index of new column names)

    Example
    -------
    >>> df, cols = create_recency_features(
    ...     df,
    ...     last_seen_col='last_seen',
    ...     date_created_col='date_created',
    ...     thresholds=[2, 7],
    ...     prefix='recency_diff_'
    ... )
    # Creates columns like:
    # - recency_diff_days (raw difference)
    # - recency_diff_<=2, recency_diff_<=7 (cumulative)
    # - recency_diff_0_to_2, recency_diff_3_to_7, recency_diff_8_plus (mutually exclusive)
    """
    df = df.copy()
    new_features = {}

    # Compute day difference
    diff_col = f"{prefix}days"
    df[diff_col] = (
        pd.to_datetime(df[last_seen_col]) - pd.to_datetime(df[date_created_col])
    ).dt.days
    new_features[diff_col] = df[diff_col]

    # Create cumulative threshold features (<=threshold)
    if include_cumulative:
        for threshold in thresholds:
            col_name = f"{prefix}<={threshold}"
            new_features[col_name] = np.where(df[diff_col] <= threshold, 1, 0)

        # Add >last_threshold feature
        last_threshold = max(thresholds)
        col_name = f"{prefix}>{last_threshold}"
        new_features[col_name] = np.where(df[diff_col] > last_threshold, 1, 0)

    # Create mutually exclusive bin features
    if include_mutually_exclusive:
        sorted_thresholds = sorted(thresholds)
        boundaries = [0] + sorted_thresholds

        for i in range(len(boundaries)):
            lower = boundaries[i]

            if i < len(boundaries) - 1:
                upper = boundaries[i + 1]
                col_name = f"{prefix}{lower}_to_{upper}"
                new_features[col_name] = np.where(
                    (df[diff_col] >= lower) & (df[diff_col] <= upper), 1, 0
                )
            else:
                # Last bin: everything above
                actual_lower = lower + 1
                col_name = f"{prefix}{actual_lower}_plus"
                new_features[col_name] = np.where(df[diff_col] >= actual_lower, 1, 0)

    # Build result DataFrame
    new_cols_df = pd.DataFrame(new_features, index=df.index)
    result_df = pd.concat(
        [df.drop(columns=[diff_col], errors="ignore"), new_cols_df], axis=1
    )
    new_cols = pd.Index(new_features.keys())

    return result_df, new_cols


def get_device_mapping_summary(
    df: pd.DataFrame,
    hardware_col: str = "ditr_hardware",
    manufacturer_col: str = "ditr_manufacturer",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate summary tables showing how raw values map to categories.

    Useful for validating the mappings and identifying unmapped values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing device columns
    hardware_col : str
        Name of hardware SKU column
    manufacturer_col : str
        Name of manufacturer column

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (hardware_summary, manufacturer_summary) - each with columns:
        original_value, category, count, pct

    Example
    -------
    >>> hw_summary, mfr_summary = get_device_mapping_summary(user_info_df)
    >>> print(hw_summary[hw_summary['category'] == 'other'])  # Check unmapped
    """
    temp_df = add_device_categories(df, hardware_col, manufacturer_col)

    # Hardware summary
    hw_summary = (
        temp_df.groupby([hardware_col, "hardware_category"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    hw_summary["pct"] = (hw_summary["count"] / hw_summary["count"].sum() * 100).round(2)
    hw_summary = hw_summary.rename(
        columns={hardware_col: "original_value", "hardware_category": "category"}
    )

    # Manufacturer summary
    mfr_summary = (
        temp_df.groupby([manufacturer_col, "manufacturer_category"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    mfr_summary["pct"] = (
        mfr_summary["count"] / mfr_summary["count"].sum() * 100
    ).round(2)
    mfr_summary = mfr_summary.rename(
        columns={
            manufacturer_col: "original_value",
            "manufacturer_category": "category",
        }
    )

    return hw_summary, mfr_summary
