from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import json
import os

from sklearn.metrics import f1_score, precision_recall_curve, auc

def evaluate_model(model, x_val, y_val):
    y_pred = model.predict(x_val)
    y_prob = model.predict_proba(x_val)[:, 1]

    # Calculate F1-Score
    f1 = f1_score(y_val, y_pred)

    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)

    report = {
        'F1-Score': f1,
        'PR-AUC': pr_auc,
        'Classification Report': classification_report(y_val, y_pred, target_names=['Not Fraud', 'Fraud'])
    }

    return report



def generate_report(reports, save_dir):
    # Summarize F1-Score and PR-AUC for each model
    summary = {
        model_name: {
            'F1-Score': report['F1-Score'],
            'PR-AUC': report['PR-AUC'],
            'Classification Report': report['Classification Report']
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
