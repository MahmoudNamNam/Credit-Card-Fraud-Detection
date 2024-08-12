# Credit-Card-Fraud-Detection


## Table of Contents

1. [Introduction](#introduction)
2. [Data Exploration and Analysis](#data-exploration-and-analysis)
   - [Loading the Data](#loading-the-data)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Feature Engineering](#feature-engineering)
3. [Modeling](#modeling)
   - [Model Selection](#model-selection)
   - [Threshold Selection](#threshold-selection)
   - [Cross-Validation](#cross-validation)
4. [Implementation](#implementation)
   - [File Structure](#file-structure)
   - [credit_fraud_train.py](#credit_fraud_trainpy)
   - [credit_fraud_utils_data.py](#credit_fraud_utils_datapy)
   - [credit_fraud_utils_eval.py](#credit_fraud_utils_evalpy)
5. [model.pkl File](#modelpkl-file)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction

This project focuses on the binary classification problem of detecting credit card fraud. The goal is to build a robust model that can accurately classify transactions as fraudulent or legitimate.

## Data Exploration and Analysis

### Loading the Data
- The dataset is loaded using Pandas.
- Initial inspection includes checking the dataset shape, column types, and summary statistics.

### Exploratory Data Analysis (EDA)
- **Missing Values:** Analyzed and handled any missing values in the dataset.
- **Data Distribution:** Visualized the distribution of key features and the target variable.
- **Class Imbalance:** Checked for imbalance in the target classes (fraud vs. non-fraud).
- **Correlation Analysis:** Investigated correlations between features.

### Feature Engineering
- **New Features:** Created additional features to enhance model performance.
- **Scaling/Normalization:** Applied scaling techniques to ensure features are on a similar scale.
- **Class Imbalance Handling:** Techniques such as SMOTE or class weighting were considered.

## Modeling

### Model Selection
- **Algorithms Used:** Tried different models such as Logistic Regression, Random Forest, and Gradient Boosting.
- **Evaluation Metrics:** Used metrics like accuracy, precision, recall, F1-score, and ROC-AUC to evaluate model performance.

### Threshold Selection
- **Best Threshold:** Selected the best classification threshold based on evaluation metrics to balance precision and recall.

### Cross-Validation
- **Cross-Validation:** Used cross-validation to ensure the model's robustness and avoid overfitting.

## Implementation

### File Structure
The project consists of the following files:
- `credit_fraud_train.py`: Main entry point for training models.
- `credit_fraud_utils_data.py`: Utility functions for data loading and processing.
- `credit_fraud_utils_eval.py`: Utility functions for model evaluation.

### `credit_fraud_train.py`
- **Purpose:** Script for training models based on user input via `argparse`.
- **Features:**
  - Loads data.
  - Trains multiple models.
  - Selects and saves the best model with the optimal threshold.

### `credit_fraud_utils_data.py`
- **Purpose:** Contains functions for data loading, cleaning, and preprocessing.
- **Key Functions:**
  - `load_data()`: Loads the dataset.
  - `preprocess_data()`: Preprocesses the data (e.g., handling missing values, scaling).

### `credit_fraud_utils_eval.py`
- **Purpose:** Contains functions for evaluating models and selecting the best threshold.
- **Key Functions:**
  - `evaluate_model()`: Evaluates the model using various metrics.
  - `find_best_threshold()`: Finds the optimal threshold for classification.

## model.pkl File
- The `model.pkl` file contains a dictionary with:
  - The trained model.
  - The best classification threshold.
  - Any other necessary information for model evaluation.

## Conclusion
- **Summary:** The project successfully identifies the best model for detecting credit card fraud, balancing precision and recall.
- **Future Work:** Possible improvements include exploring additional features or using advanced ensemble methods.

## References
- List any references or resources used in the project.
