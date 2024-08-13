# Credit-Card-Fraud-Detection

## Table of Contents

1. [Introduction](#introduction)
2. [Data Exploration and Analysis](#data-exploration-and-analysis)
   - [Loading the Data](#loading-the-data)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Feature Engineering](#feature-engineering)
3. [Modeling](#modeling)
   - [Model Selection](#model-selection)
   - [Resampling Techniques](#resampling-techniques)
   - [Threshold Selection](#threshold-selection)
   - [Cross-Validation](#cross-validation)
4. [Implementation](#implementation)
   - [File Structure](#file-structure)
   - [credit_fraud_train.py](#credit_fraud_trainpy)
   - [credit_fraud_test.py](#credit_fraud_testpy)
   - [credit_fraud_utils_data.py](#credit_fraud_utils_datapy)
   - [credit_fraud_utils_eval.py](#credit_fraud_utils_evalpy)
5. [Reports](#reports)
6. [model.pkl File](#modelpkl-file)
7. [Conclusion](#conclusion)

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

- **Scaling/Normalization:** Applied scaling techniques to ensure features are on a similar scale.
- **Class Imbalance Handling:** Techniques such as SMOTE, undersampling, or oversampling were applied to address class imbalance.

## Modeling

### Model Selection

- **Algorithms Used:** Evaluated models including Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, Neural Networks, and Voting Classifiers.
- **Evaluation Metrics:** Metrics such as accuracy, precision, recall, F1-score, PR-AUC, and ROC-AUC were used to assess model performance.

### Resampling Techniques

- **Techniques:** Resampling techniques such as SMOTE, undersampling, and oversampling were applied to address class imbalance.
- **Reports:** Detailed reports for each resampling technique (SMOTE, undersampling, oversampling) are saved in the `Report` folder.

### Threshold Selection

- **Best Threshold:** The best classification threshold was selected based on F1-score and other evaluation metrics to balance precision and recall.

### Cross-Validation

- **Cross-Validation:** Cross-validation was used to validate the models' performance and to avoid overfitting.

## Implementation

### File Structure

The project consists of the following files:

- `credit_fraud_train.py`: Main script for training models based on user input via `argparse`.
- `credit_fraud_test.py`: Script for testing the trained model on a test dataset.
- `credit_fraud_utils_data.py`: Utility functions for data loading and preprocessing.
- `credit_fraud_utils_eval.py`: Utility functions for model evaluation and threshold selection.

### `credit_fraud_train.py`

- **Purpose:** Script for training multiple models and selecting the best one based on evaluation metrics.
- **Features:**
  - Loads and preprocesses training and validation data.
  - Applies resampling techniques.
  - Trains models.
  - Evaluates models and saves the best-performing model along with the optimal threshold.

### `credit_fraud_test.py`

- **Purpose:** Script for testing the saved model on a test dataset.
- **Features:**
  - Loads and preprocesses test data.
  - Loads the trained model and applies it to the test data.
  - Generates evaluation reports including classification metrics and ROC-AUC score.

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
  - `generate_report()`: Generates detailed reports for each model and resampling technique.

## Reports

- **Location:** Detailed reports for each resampling technique (SMOTE, undersampling, oversampling) are stored in the `Report` folder.
- **Content:** Each report includes metrics such as F1-score, PR-AUC, and the best threshold for different models.

## models

- The `model.pkl` file contains a dictionary with:
  - The trained model.
  - The best classification threshold.
  - Any other necessary information for model evaluation.

## Conclusion

- **Summary:** The project successfully identifies the best model for detecting credit card fraud, balancing precision and recall across various resampling techniques.
- **Future Work:** Possible improvements include exploring additional features, advanced ensemble methods, and real-time fraud detection.
