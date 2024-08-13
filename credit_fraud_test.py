import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from credit_fraud_utils_data import load_and_preprocess_data

def test_model(test_data_path, model_path):
    # Load the test data
    test = load_and_preprocess_data(test_data_path)
    test_np = test.to_numpy()
    x_test, y_test = test_np[:, :-1], test_np[:, -1]

    # Load the model
    model = joblib.load(model_path)

    # Evaluate the model
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'])
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Classification Report:\n{report}")
    print(f"ROC-AUC Score: {roc_auc}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the saved model on test data")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to the test data CSV file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model file")
    
    args = parser.parse_args()
    test_model(args.test_data_path, args.model_path)
