import argparse
import pickle
import os
from credit_fraud_utils_data import load_and_preprocess_data, apply_resampling_techniques
from credit_fraud_utils_eval import evaluate_model, generate_report, find_best_threshold,save_model_with_threshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def main(args):
    # Load and preprocess training data
    train = load_and_preprocess_data(args.train_data_path)
    train_np = train.to_numpy()
    x_train, y_train = train_np[:, :-1], train_np[:, -1]
    x_train, y_train = apply_resampling_techniques(x_train, y_train, method=args.resampling_method)

    # Load and preprocess validation data
    val = load_and_preprocess_data(args.val_data_path)
    val_np = val.to_numpy()
    x_val, y_val = val_np[:, :-1], val_np[:, -1]
    
    # Define models
    logistic_model = LogisticRegression(class_weight='balanced')
    random_forest = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    gradient_boosting = GradientBoostingClassifier()
    ada_boost = AdaBoostClassifier()
    neural_network = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=50, random_state=42, early_stopping=True)

    # Voting Classifier (combination of best models: Logistic Regression and Random Forest)
    voting_model = VotingClassifier(estimators=[
        ('lr', logistic_model),
        ('rf', random_forest)
    ], voting='soft')

    models = {
        'Logistic Regression': logistic_model,
        'Random Forest': random_forest,
        'Gradient Boosting': gradient_boosting,
        'AdaBoost': ada_boost,
        'Neural Network': neural_network,
        'Voting Classifier': voting_model
    }

    reports = {}
    os.makedirs(args.model_save_dir, exist_ok=True)

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        
        # Evaluate the model on the validation set
        report = evaluate_model(model, x_val, y_val)
        
        # Find the best threshold and update report with it
        best_threshold, best_f1 = find_best_threshold(model, x_val, y_val)
        report['Best Threshold'] = best_threshold
        report['F1-Score (Best Threshold)'] = best_f1
        
        # Save model with best threshold using the new function
        save_model_with_threshold(model, best_threshold, model_name, args.model_save_dir)
        
        reports[model_name] = report

    generate_report(reports, args.model_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for Credit Card Fraud Detection")
    parser.add_argument('--train_data_path', type=str, default='./data/train.csv', help="Path to the training data CSV file")
    parser.add_argument('--val_data_path', type=str, default='./data/val.csv', help="Path to the validation data CSV file")
    parser.add_argument('--resampling_method', type=str, required=True, choices=['undersample', 'smote', 'oversample'], help="Resampling method")
    parser.add_argument('--model_save_dir', type=str, default='./models/', help="Directory to save the trained models and reports")
    
    args = parser.parse_args()
    main(args)
