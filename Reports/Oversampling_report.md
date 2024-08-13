# **Credit Card Fraud Detection Model Evaluation Report (Oversampling)**

## **1. Logistic Regression**

- **F1-Score:** 0.0968
- **Precision-Recall AUC (PR-AUC):** 0.7598
- **Confusion Matrix:**
  - True Negative: 55,387
  - False Positive: 1,483
  - False Negative: 10
  - True Positive: 80
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.05 (Fraud)
  - Recall: 0.97 (Not Fraud), 0.89 (Fraud)
  - Accuracy: 97%
- **Best Threshold:** 0.99
- **F1-Score (Best Threshold):** 0.6698

**Observation:**
Logistic Regression shows strong recall for fraud cases but suffers from low precision, leading to many false positives. The model's overall F1-Score is low, reflecting its challenges in handling imbalanced data.

---

## **2. Random Forest**

- **F1-Score:** 0.8176
- **Precision-Recall AUC (PR-AUC):** 0.8582
- **Confusion Matrix:**
  - True Negative: 56,866
  - False Positive: 4
  - False Negative: 25
  - True Positive: 65
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.94 (Fraud)
  - Recall: 1.00 (Not Fraud), 0.72 (Fraud)
  - Accuracy: 100%
- **Best Threshold:** 0.34
- **F1-Score (Best Threshold):** 0.8588

**Observation:**
Random Forest performs exceptionally well, with a high F1-Score and PR-AUC. It effectively balances precision and recall, making it a reliable model for fraud detection in this dataset.

---

## **3. Gradient Boosting**

- **F1-Score:** 0.3286
- **Precision-Recall AUC (PR-AUC):** 0.7650
- **Confusion Matrix:**
  - True Negative: 56,548
  - False Positive: 322
  - False Negative: 9
  - True Positive: 81
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.20 (Fraud)
  - Recall: 0.99 (Not Fraud), 0.90 (Fraud)
  - Accuracy: 99%
- **Best Threshold:** 0.96
- **F1-Score (Best Threshold):** 0.8182

**Observation:**
Gradient Boosting shows moderate performance with a balanced recall for fraud detection. However, its low precision impacts its overall F1-Score, indicating potential issues with false positives.

---

## **4. AdaBoost**

- **F1-Score:** 0.1479
- **Precision-Recall AUC (PR-AUC):** 0.7645
- **Confusion Matrix:**
  - True Negative: 55,971
  - False Positive: 899
  - False Negative: 11
  - True Positive: 79
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.08 (Fraud)
  - Recall: 0.98 (Not Fraud), 0.88 (Fraud)
  - Accuracy: 98%
- **Best Threshold:** 0.53
- **F1-Score (Best Threshold):** 0.7692

**Observation:**
AdaBoost offers high recall but low precision, resulting in many false positives. While the model can detect fraud cases, its low F1-Score suggests challenges in handling imbalanced datasets effectively.

---

## **5. Neural Network**

- **F1-Score:** 0.7784
- **Precision-Recall AUC (PR-AUC):** 0.7950
- **Confusion Matrix:**
  - True Negative: 56,847
  - False Positive: 23
  - False Negative: 18
  - True Positive: 72
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.76 (Fraud)
  - Recall: 1.00 (Not Fraud), 0.80 (Fraud)
  - Accuracy: 100%
- **Best Threshold:** 0.99
- **F1-Score (Best Threshold):** 0.8095

**Observation:**
The Neural Network model provides a good balance between precision and recall for fraud detection, with a high F1-Score and PR-AUC. Its ability to minimize false positives while maintaining decent recall makes it a strong candidate for deployment.

---

## **6. Voting Classifier**

- **F1-Score:** 0.6916
- **Precision-Recall AUC (PR-AUC):** 0.8256
- **Confusion Matrix:**
  - True Negative: 56,820
  - False Positive: 50
  - False Negative: 16
  - True Positive: 74
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.60 (Fraud)
  - Recall: 1.00 (Not Fraud), 0.82 (Fraud)
  - Accuracy: 100%
- **Best Threshold:** 0.63
- **F1-Score (Best Threshold):** 0.8503

**Observation:**
The Voting Classifier shows strong performance with a high F1-Score and PR-AUC. It effectively balances precision and recall, making it a robust model for fraud detection. The Voting Classifier performs well across various metrics, showing the benefits of combining multiple models.

---

## **Summary and Recommendations**

- **Best Overall Model:** Random Forest and the Voting Classifier are top performers, with high F1-Scores and PR-AUC. Both models are well-suited for fraud detection, offering a good balance between precision and recall.

- **High Recall Models:** The Neural Network model also demonstrates strong recall, essential for minimizing false negatives in fraud detection.

- **Precision vs. Recall Trade-off:** Models like AdaBoost and Logistic Regression emphasize recall but at the cost of low precision, leading to higher false positive rates.

- **Threshold Tuning:** Fine-tuning decision thresholds can significantly enhance model performance, especially in imbalanced datasets. Random Forest and Voting Classifier benefit from threshold optimization.
