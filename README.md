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
xxx
```

### EDA and Feature Engineering

```python
xxx
```

### Model Training

```python
xxx
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


## Contact

