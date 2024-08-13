from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os
from sklearn.metrics import f1_score, precision_recall_curve, auc, accuracy_score
import pickle

def evaluate_model(model, x_val, y_val):
    y_pred = model.predict(x_val)
    y_prob = model.predict_proba(x_val)[:, 1]

    # Calculate F1-Score
    f1 = f1_score(y_val, y_pred)

    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)

    # Calculate Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    cm_dict = {
        "True Negative": int(cm[0][0]),
        "False Positive": int(cm[0][1]),
        "False Negative": int(cm[1][0]),
        "True Positive": int(cm[1][1])
    }

    report = {
        'F1-Score': f1,
        'PR-AUC': pr_auc,
        'Confusion Matrix': cm_dict,
        'Classification Report': classification_report(y_val, y_pred, target_names=['Not Fraud', 'Fraud'])
    }

    return report


def generate_report(reports, save_dir):
    # Summarize F1-Score, PR-AUC, and Confusion Matrix for each model
    summary = {
        model_name: {
            'F1-Score': report['F1-Score'],
            'PR-AUC': report['PR-AUC'],
            'Confusion Matrix': report['Confusion Matrix'],
            'Classification Report': report['Classification Report'],
            'Best Threshold': report.get('Best Threshold', None)
        }
        for model_name, report in reports.items()
    }

    # Save the detailed reports
    report_file = os.path.join(save_dir, 'model_reports.json')
    with open(report_file, 'w') as f:
        json.dump(reports, f, indent=4)
    print(f"Detailed model reports saved to '{report_file}'")

    # Save the summary
    summary_file = os.path.join(save_dir, 'model_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Model summary saved to '{summary_file}'")


def find_best_threshold(model, x_val, y_val):
    """
    Find the best threshold for the model that optimizes the F1-score.

    Parameters:
    - model: Trained model (must have a predict_proba method)
    - X_val: Validation features
    - y_val: True labels for the validation set

    Returns:
    - best_threshold: The threshold that yields the highest F1-score
    - best_f1: The best F1-score obtained
    """

    # Get predicted probabilities for the positive class (fraud)
    y_probs = model.predict_proba(x_val)[:, 1]

    # Initialize variables to store the best threshold and best F1-score
    best_threshold = 0.0
    best_f1 = 0.0

    # Evaluate thresholds from 0.01 to 0.99
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def save_model_with_threshold(model, best_threshold, model_name, save_dir):
    """
    Save the trained model along with the best threshold and model name to a pickle file.

    Parameters:
    - model: The trained model to save.
    - best_threshold: The best threshold for classification.
    - model_name: The name of the model.
    - save_dir: Directory where the model will be saved.
    
    Returns:
    - model_file: The path to the saved model file.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create the model dictionary
    model_dict = {
        "model": model,
        "threshold": best_threshold,
        "model_name": model_name
    }

    # Define the path for saving the model
    model_file = os.path.join(save_dir, f'{model_name.replace(" ", "_").lower()}_model.pkl')

    # Save the model to a pickle file
    with open(model_file, 'wb') as file:
        pickle.dump(model_dict, file)
    
    print(f"{model_name} model with best threshold saved to '{model_file}'")

    return model_file
