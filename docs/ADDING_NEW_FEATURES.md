# Adding New Features to the Pipeline

This guide explains how to add new features to the wonky study analysis pipeline. The pipeline follows a 4-step pattern, and adding a new feature typically requires modifications to multiple files.

---

## Running Concurrently (Multiple Users)

When multiple people run the pipeline simultaneously, use the `PipelineConfig` class to namespace your output files and avoid conflicts.

### Quick Start

```python
# At the top of each notebook, instead of loading paths manually:
from src.config import PipelineConfig

# Set YOUR identifier (use your name or a unique run ID)
cfg = PipelineConfig(run_id="dan")  # or "alex", "run_20250204", etc.

# Access paths
silver_path = cfg.silver_path
output_path = cfg.get_output_path("user_info_df")
# Returns: "dbfs:/FileStore/misc/dan/user_info_df_pullcomplete.parquet"

# Get all output paths at once
output_paths = cfg.output_paths
```

### How It Works

1. All output files are namespaced by `run_id` in the path
2. Input tables (silver/bronze) are shared and read-only
3. Each user's outputs go to their own folder: `dbfs:/FileStore/misc/{run_id}/`

### Alternative: Set in Config File

You can also set the default `run_id` in `configs/data_paths.yaml`:

```yaml
# Set this to your name/identifier
run_id: "dan"
```

Then just use `PipelineConfig()` without arguments.

---

## Pipeline Overview

```
Step 1: Data Pull (Notebook 1)
    ├── Load data from Delta tables (loaders.py)
    └── Join tables together (joiners.py)
           ↓
Step 2: Feature Engineering (Notebook 2)
    └── Create engineered features (feature_engineering.py, feature_engineering_utils.py)
           ↓
Step 3: Statistical Testing (Notebook 3)
    └── Run OLS/Logistic regression on feature sets (statistical_tests.yaml)
           ↓
Step 4: Modelling (Notebook 4)
    └── Random Forest + SHAP analysis
```

---

## Quick Reference: Files to Modify

| Step | File | Purpose |
|------|------|---------|
| 1 | `src/data_pull/loaders.py` | Load data from new table |
| 2 | `src/data_pull/joiners.py` | Join new data to main dataframe |
| 3 | `configs/data_paths.yaml` | Register new table name |
| 4 | `notebooks/1.respondent_datapull_processing.ipynb` | Call loader and joiner |
| 5 | `notebooks/2.respondent_feature_engineering.ipynb` | Engineer features (e.g., one-hot encode) |
| 6 | `configs/statistical_tests.yaml` | Register feature set for testing |

---

## Step-by-Step Guide

### Step 1: Add Data Loader

**File:** `src/data_pull/loaders.py`

Add a function to load your new data source. Follow the existing pattern:

```python
def load_your_new_table(
    spark: SparkSession,
    silver_path: str,
    select_cols: Optional[List[str]] = None,
) -> DataFrame:
    """
    Load your_new_table from silver layer.

    Parameters
    ----------
    spark : SparkSession
    silver_path : str
        Path to silver layer
    select_cols : list, optional
        Columns to select

    Returns
    -------
    DataFrame
        Your new data
    """
    if select_cols is None:
        select_cols = ["respondent_pk", "your_column"]

    df = spark.read.format("delta").load(f"{silver_path}your_new_table")

    available_cols = [c for c in select_cols if c in df.columns]
    if available_cols:
        df = df.select(*available_cols)

    return df
```

**Key patterns:**
- Always accept `spark` and `silver_path` as parameters
- Use optional `select_cols` to allow column filtering
- Check that columns exist before selecting
- Return a Spark DataFrame

---

### Step 2: Add Join Function

**File:** `src/data_pull/joiners.py`

Add a function to join your new data to the main dataframe:

```python
def join_your_new_table(
    main_df: DataFrame,
    new_df: DataFrame,
    main_join_col: str = "respondent_pk",
    new_join_col: str = "respondent_pk",
    column_alias: str = "your_new_column",
) -> DataFrame:
    """
    Join your new table data to main dataframe.

    Parameters
    ----------
    main_df : DataFrame
        Main Spark DataFrame to join to
    new_df : DataFrame
        Your new Spark DataFrame
    main_join_col : str
        Column name in main_df to join on
    new_join_col : str
        Column name in new_df to join on
    column_alias : str
        Alias for the column in output

    Returns
    -------
    DataFrame
        Joined DataFrame with new column added
    """
    # Select only needed columns and alias to avoid conflicts
    new_cols = new_df.select(
        col(new_join_col).alias("_temp_join_key"),
        col("your_column").alias(column_alias),
    )

    # Left join to preserve all records (allows nulls)
    result_df = main_df.join(
        new_cols,
        main_df[main_join_col] == new_cols["_temp_join_key"],
        "left",
    ).drop("_temp_join_key")

    return result_df
```

**Key patterns:**
- Use `left` join to preserve all records (allows null values)
- Alias columns to avoid naming conflicts
- Drop temporary join keys after joining

---

### Step 3: Register in Configuration

**File:** `configs/data_paths.yaml`

Add your new table to the tables section:

```yaml
tables:
  user: "user"
  task_complete: "task_complete"
  respondent_info: "respondent_info"
  task: "task"
  your_new_table: "your_new_table"  # Add this line
```

---

### Step 4: Use in Notebook 1 (Data Pull)

**File:** `notebooks/1.respondent_datapull_processing.ipynb`

Add imports and calls to load and join your data:

```python
# Import the new functions
from src.data_pull.loaders import load_your_new_table
from src.data_pull.joiners import join_your_new_table

# Load the new data
new_df = load_your_new_table(
    spark,
    silver_path,
    select_cols=["respondent_pk", "your_column"]
)

# Join to main dataframe (after existing joins)
user_info_spark = join_your_new_table(
    user_info_spark,
    new_df,
    main_join_col="respondent_pk"
)
```

---

### Step 5: Engineer Features in Notebook 2

**File:** `notebooks/2.respondent_feature_engineering.ipynb`

For categorical features, use the existing one-hot encoding utility:

```python
from src.eda.feature_engineering_utils import one_hot_encode_column

# One-hot encode your new column (handles nulls automatically)
df, new_feature_cols = one_hot_encode_column(
    df,
    column='your_new_column',
    prefix='your_new_column'
)

# This creates columns like:
# - your_new_column_value1
# - your_new_column_value2
# - your_new_column_nan (for null values)
```

For numeric features, you can create threshold-based features:

```python
from src.eda.feature_engineering_utils import create_binned_features

df, bin_cols = create_binned_features(
    df,
    column='your_numeric_column',
    bins=[10, 25, 50, 100],
    prefix='your_column'
)
```

---

### Step 6: Register Feature Set for Testing

**File:** `configs/statistical_tests.yaml`

Add your new feature set to enable statistical testing:

```yaml
feature_sets:
  # ... existing feature sets ...

  your_new_feature:
    - your_new_column_value1
    - your_new_column_value2
    - your_new_column_value3
    - your_new_column_nan
```

**Note:** After running feature engineering for the first time, check the actual column names created and update this list accordingly.

---

## Example: Adding `stripe_country` Feature

Here's a complete example of adding the `stripe_country` feature from the `stripe_verification` table:

### 1. Loader (`src/data_pull/loaders.py`)

```python
def load_stripe_verification(
    spark: SparkSession,
    silver_path: str,
    select_cols: Optional[List[str]] = None,
) -> DataFrame:
    if select_cols is None:
        select_cols = ["respondent_pk", "country"]

    df = spark.read.format("delta").load(f"{silver_path}stripe_verification")

    available_cols = [c for c in select_cols if c in df.columns]
    if available_cols:
        df = df.select(*available_cols)

    return df
```

### 2. Joiner (`src/data_pull/joiners.py`)

```python
def join_stripe_verification(
    main_df: DataFrame,
    stripe_df: DataFrame,
    main_join_col: str = "respondent_pk",
    stripe_join_col: str = "respondent_pk",
    country_alias: str = "stripe_country",
) -> DataFrame:
    stripe_cols = stripe_df.select(
        col(stripe_join_col).alias("_stripe_respondent_pk"),
        col("country").alias(country_alias),
    )

    result_df = main_df.join(
        stripe_cols,
        main_df[main_join_col] == stripe_cols["_stripe_respondent_pk"],
        "left",
    ).drop("_stripe_respondent_pk")

    return result_df
```

### 3. Config (`configs/data_paths.yaml`)

```yaml
tables:
  # ... existing tables ...
  stripe_verification: "stripe_verification"
```

### 4. Notebook 1 Usage

```python
from src.data_pull.loaders import load_stripe_verification
from src.data_pull.joiners import join_stripe_verification

# Load stripe data
stripe_df = load_stripe_verification(spark, silver_path)

# Join to main dataframe
user_info_spark = join_stripe_verification(user_info_spark, stripe_df)
```

### 5. Notebook 2 Feature Engineering

```python
from src.eda.feature_engineering_utils import one_hot_encode_column

df, stripe_cols = one_hot_encode_column(df, 'stripe_country', prefix='stripe_country')
```

### 6. Statistical Tests Config (`configs/statistical_tests.yaml`)

```yaml
  stripe_country:
    - stripe_country_GB
    - stripe_country_US
    - stripe_country_DE
    - stripe_country_FR
    - stripe_country_nan
```

---

## Common Patterns

### Handling Nullable Columns

When a feature can be null (like `stripe_country`):
- Use `left` join to preserve all records
- One-hot encoding automatically creates a `_nan` category
- No special null handling needed in most cases

### Reducing Cardinality

For high-cardinality columns (many unique values), consider:
1. **Value mapping:** Group similar values (see `map_hardware_to_category()` in `feature_engineering_utils.py`)
2. **Binning:** Convert to ranges (see `create_binned_features()`)
3. **Top-N encoding:** Keep top N values, group rest as "other"

### Column Naming Conventions

- Prefix engineered features with the source column name
- Use underscores, not hyphens
- For one-hot encoded columns: `{prefix}_{value}`
- For binned columns: `{prefix}_{bin_label}`

---

## Checklist

- [ ] Added loader function in `src/data_pull/loaders.py`
- [ ] Added joiner function in `src/data_pull/joiners.py`
- [ ] Added table to `configs/data_paths.yaml`
- [ ] Added data loading to Notebook 1
- [ ] Added feature engineering to Notebook 2
- [ ] Added feature set to `configs/statistical_tests.yaml`
- [ ] Tested pipeline end-to-end
- [ ] Updated feature list in YAML after seeing actual values

---

## Configuration Reference

### `PipelineConfig` Class

The `src/config.py` module provides a `PipelineConfig` class for managing paths:

```python
from src.config import PipelineConfig

# Initialize with your run_id
cfg = PipelineConfig(run_id="your_name")

# Available properties (read-only, shared):
cfg.silver_path          # "/mnt/delta/silver/"
cfg.bronze_path          # "/mnt/delta/bronze/"
cfg.gold_path            # "/mnt/delta/gold/"
cfg.project_repository_path
cfg.tables               # dict of table names
cfg.filters              # dict of data filters

# Output paths (namespaced by run_id):
cfg.get_output_path("user_info_df")      # Single path
cfg.get_output_path("test_results_df")
cfg.output_paths                          # All paths as dict

# Check current run_id
print(cfg.run_id)        # "your_name"
```

### Output Files

| Key | Description |
|-----|-------------|
| `user_info_df` | Merged user/task data from Notebook 1 |
| `user_info_df_post_eda` | Feature-engineered data from Notebook 2 |
| `test_results_df` | Statistical test results from Notebook 3 |
| `feature_summary` | Model feature importance from Notebook 4 |
| `interactions` | SHAP interaction values |
| `*_control` / `*_exposed` | Stratified analysis outputs |
