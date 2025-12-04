# onDeviceResearch Column Redacts

A comprehensive fraud detection and data quality improvement system for on-device research studies.

## Overview

This repository contains tools and models for:
- **Data Pulling**: Extract and process data from Delta tables using PySpark
- **Exploratory Data Analysis**: Statistical tests and feature engineering
- **Fraud Detection**: Unsupervised anomaly detection, behavioral clustering, and supervised prediction models
- **Data Quality Improvement**: Cluster-based imputation for missing values

## Workflow

```
Data Pull → EDA & Statistical Modelling → Model Training → Fraud Detection & Data Quality Improvement
```

The outputs from EDA and statistical tests inform:
- Which features to use in models
- Which model types are most appropriate
- Feature selection strategies
- Imputation strategies based on cluster groups

## Model Objectives

1. **Fraud Rate Reduction**: Target reduction to ~5% fraud rate
2. **Data Quality Improvement**: Fill nulls using cluster-based predictions
3. **Fraud Detection**: Predict likelihood of fraud using supervised models
4. **Behavioral Insights**: Use clustering to identify behavioral patterns
5. **Anomaly Detection**: Identify outliers and anomalous behavior patterns

## Repository Structure

```
onDeviceResearch_columnRedacts/
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code modules
│   ├── data_pull/         # Data loading and processing
│   ├── eda/               # Exploratory data analysis
│   ├── models/            # ML models
│   ├── preprocessing/     # Data preprocessing
│   └── utils/             # Utility functions
├── configs/               # Configuration files (YAML)
└── requirements.txt       # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure paths in `configs/data_paths.yaml`

3. Update model parameters in `configs/models.yaml` if needed

## Usage

### Data Pulling

```python
from src.data_pull.loaders import load_user_table, load_task_complete_table
from src.data_pull.joiners import join_user_task_respondent

# Load data
user_df = load_user_table(spark, silver_path, country="GB")
task_df = load_task_complete_table(spark, silver_path, min_date="2025-10-10")
respondent_df = load_respondent_info_table(spark, silver_path)

# Join tables
joined_df = join_user_task_respondent(user_df, task_df, respondent_df)
```

### EDA and Feature Engineering

```python
from src.eda.feature_engineering import create_time_features, create_respondent_behavioral_features
from src.eda.statistical_tests import compare_groups_statistically

# Create features
df_with_time = create_time_features(df, date_col="date_completed")
respondent_features = create_respondent_behavioral_features(df_with_time)

# Statistical tests
results = compare_groups_statistically(
    respondent_features,
    group_col="has_wonky_tasks",
    metrics=["avg_task_time", "suspicious_fast_rate"]
)
```

### Model Training

```python
from src.models.supervised import train_fraud_model
from src.models.model_evaluation import evaluate_model_performance

# Train model
model, X_train, X_test, y_train, y_test = train_fraud_model(
    X, y, method="random_forest"
)

# Evaluate
metrics = evaluate_model_performance(
    y_test, y_pred, y_proba, baseline_fraud_rate=0.15, target_fraud_rate=0.05
)
```

## Configuration

All configuration is done through YAML files in the `configs/` directory:

- `data_paths.yaml`: Data paths and table names
- `feature_engineering.yaml`: Feature engineering parameters
- `statistical_tests.yaml`: Statistical test settings
- `wonky_studies.yaml`: Wonky study UUIDs and column mappings
- `models.yaml`: Model hyperparameters and evaluation settings
- `preprocessing.yaml`: Imputation and preprocessing settings

## Databricks Integration

This repository is configured for Databricks with `databricks.yml`. To connect:

1. Configure Databricks CLI
2. Sync the repository to your Databricks workspace
3. Run notebooks in Databricks environment

## License

[Add your license here]

## Contact

[Add contact information here]

