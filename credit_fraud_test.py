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
    model_dict = joblib.load(model_path)
    model = model_dict['model']
    best_threshold = model_dict['threshold']

    # Evaluate the model using the best threshold
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    # Generate and print the classification report and ROC-AUC score
    report = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'])
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Classification Report:\n{report}")
    print(f"ROC-AUC Score: {roc_auc}")

    # Optionally, save the results to a CSV for further analysis
    results = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob})
    results.to_csv('test_results.csv', index=False)
    print("Test results saved to 'test_results.csv'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the saved model on test data")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to the test data CSV file")
    parser.add_argument('--model_path', type=str, default='models/Smote/random_forest_model.pkl', help="Path to the saved model file")
    
    args = parser.parse_args()
    test_model(args.test_data_path, args.model_path)
