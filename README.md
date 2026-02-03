# Wonkiness Analysis Pipeline

This codebase provides a pipeline for measuring and reporting on wonkiness of studies in balance tables.

## Overview

The analysis follows a four-step workflow, with each step handled by a dedicated notebook:

```
┌─────────────────────────────────┐
│  1. Data Pull & Processing      │
│  (respondent_datapull_          │
│   processing.ipynb)             │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  2. Feature Engineering         │
│  (respondent_feature_           │
│   engineering.ipynb)            │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  3. Statistical Testing         │
│  (respondent_testing.ipynb)     │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  4. Modelling                   │
│  (respondent_modelling.ipynb)   │
└─────────────────────────────────┘
```

---

## Step 1: Data Pull & Processing

**Notebook:** `notebooks/1.respondent_datapull_processing.ipynb`

This step loads and preprocesses the raw data required for analysis.

### Supporting Files

| File | Purpose |
|------|---------|
| `src/data_pull/loaders.py` | Data loading functions |
| `src/data_pull/aggregators.py` | Data aggregation utilities |
| `src/data_pull/joiners.py` | Data joining and merging logic |
| `configs/data_paths.yaml` | Data source path configurations |
| `configs/wonky_studies.yaml` | Wonky study definitions |

### Output
Processed dataset ready for feature engineering.

---

## Step 2: Feature Engineering

**Notebook:** `notebooks/2.respondent_feature_engineering.ipynb`

This step creates features from the processed data that will be used in statistical testing and modelling.

### Supporting Files

| File | Purpose |
|------|---------|
| `src/eda/feature_engineering.py` | Main feature engineering functions |
| `src/eda/feature_engineering_utils.py` | Utility functions for feature creation |

### Output
Feature-engineered and one-hot-encodiig dataset to prepare features for testing and modelling.


---

## Step 3: Statistical Testing
**Notebook:** `notebooks/3.respondent_testing.ipynb`


This step performs statistical tests to analyze relationships and validate hypotheses about wonkiness.

Uses a combination of OLS and Logistic regression, clustered at a user level to avoid biases skewed to wards
high volume taskers.

### Supporting Files


| File | Purpose |
|------|---------|
| `src/eda/statistical_tests.py` | Statistical testing functions |
| `src/eda/visualizations.py` | Visualization utilities |
| `configs/statistical_tests.yaml` | Test configurations and parameters |

### Output
Statistical test results and visualizations.

---

## Step 4: Modelling

**Notebook:** `notebooks/4.respondent_modelling.ipynb`

This step builds and evaluates models to explain wonkiness in studies.

### Supporting Files


| File | Purpose |
|------|---------|
| `src/modelling/modelling.py` | Core modelling functions |
| `src/modelling/modelling_utils.py` | Model utility functions |
| `src/modelling/modelling_visualization.py` | Model visualization utilities |
| `configs/models.yaml` | Model configurations and hyperparameters |


### Output

Trained models, feature importance rankings, and reporting artifacts.


---

## Project Structure

```
├── configs/                    # Configuration files
│   ├── data_paths.yaml
│   ├── models.yaml
│   ├── statistical_tests.yaml
│   └── wonky_studies.yaml
├── notebooks/                  # Analysis notebooks
│   ├── 1.respondent_datapull_processing.ipynb
│   ├── 2.respondent_feature_engineering.ipynb
│   ├── 3.respondent_testing.ipynb
│   └── 4.respondent_modelling.ipynb
└── src/                        # Source code
    ├── data_pull/              # Data loading modules
    ├── eda/                    # EDA and feature engineering
    └── modelling/              # Modelling modules
```

---

## Usage

Run the notebooks in order (1 → 2 → 3 → 4) to complete the full analysis pipeline. Each notebook builds on the outputs of the previous step.
