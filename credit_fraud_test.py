import joblib
import pandas as pd
from credit_fraud_utils_data import load_and_preprocess_data
from credit_fraud_utils_eval import evaluate_model

def main(test_data_path, model_path):
    # Load and preprocess test data
    test = load_and_preprocess_data(test_data_path)
    test_np = test.to_numpy()
    x_test, y_test = test_np[:, :-1], test_np[:, -1]
    
    # Load the saved model
    model = joblib.load(model_path)
    
    # Evaluate the model on the test set
    evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the trained model on a test dataset")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to the test data CSV file")
    parser.add_argument('--model_path', type=str, default='./models/best_model.pkl', help="Path to the saved model file")
    
    args = parser.parse_args()
    main(args.test_data_path, args.model_path)
