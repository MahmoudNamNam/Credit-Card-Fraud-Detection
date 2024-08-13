# **Credit Card Fraud Detection Model Evaluation Report (Smote)**

## **1. AdaBoost**

- **F1-Score:** 0.1155
- **Precision-Recall AUC (PR-AUC):** 0.7871
- **Confusion Matrix:**
  - True Negative: 55,606
  - False Positive: 1,264
  - False Negative: 7
  - True Positive: 83
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.06 (Fraud)
  - Recall: 0.98 (Not Fraud), 0.92 (Fraud)
  - Accuracy: 98%
- **Best Threshold:** 0.53
- **Best F1-Score:** 0.7590

**Observation:**
AdaBoost shows strong recall for detecting fraud cases (0.92) but suffers from low precision (0.06), indicating many false positives. The F1-Score is relatively low, highlighting the model’s struggle with imbalanced data.

---

## **2. Gradient Boosting**

- **F1-Score:** 0.1924
- **Precision-Recall AUC (PR-AUC):** 0.7565
- **Confusion Matrix:**
  - True Negative: 56,199
  - False Positive: 671
  - False Negative: 9
  - True Positive: 81
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.11 (Fraud)
  - Recall: 0.99 (Not Fraud), 0.90 (Fraud)
  - Accuracy: 99%
- **Best Threshold:** 0.96
- **Best F1-Score:** 0.8295

**Observation:**
Gradient Boosting offers a better balance between precision and recall for fraud detection compared to AdaBoost. It has a higher F1-Score, suggesting improved performance, though it still struggles with precision.

---

## **3. Logistic Regression**

- **F1-Score:** 0.0942
- **Precision-Recall AUC (PR-AUC):** 0.7614
- **Confusion Matrix:**
  - True Negative: 55,341
  - False Positive: 1,529
  - False Negative: 10
  - True Positive: 80
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.05 (Fraud)
  - Recall: 0.97 (Not Fraud), 0.89 (Fraud)
  - Accuracy: 97%
- **Best Threshold:** 0.99
- **Best F1-Score:** 0.6520

**Observation:**
Logistic Regression exhibits lower F1-Score and struggles with precision for fraud cases. While recall is high, the model generates many false positives, which impacts its practical utility.

---

## **4. Neural Network**

- **F1-Score:** 0.7692
- **Precision-Recall AUC (PR-AUC):** 0.7763
- **Confusion Matrix:**
  - True Negative: 56,848
  - False Positive: 22
  - False Negative: 20
  - True Positive: 70
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.76 (Fraud)
  - Recall: 1.00 (Not Fraud), 0.78 (Fraud)
  - Accuracy: 100%
- **Best Threshold:** 0.99
- **Best F1-Score:** 0.7875

**Observation:**
The Neural Network model provides a strong balance between precision and recall for fraud detection, achieving the highest accuracy and a high F1-Score. Its ability to minimize false positives while maintaining decent recall makes it a reliable choice.

---

## **5. Random Forest**

- **F1-Score:** 0.8284
- **Precision-Recall AUC (PR-AUC):** 0.8437
- **Confusion Matrix:**
  - True Negative: 56,861
  - False Positive: 9
  - False Negative: 20
  - True Positive: 70
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.89 (Fraud)
  - Recall: 1.00 (Not Fraud), 0.78 (Fraud)
  - Accuracy: 100%
- **Best Threshold:** 0.47
- **Best F1-Score:** 0.8421

**Observation:**
Random Forest stands out with its high F1-Score and Precision-Recall AUC. It demonstrates excellent precision and recall, making it one of the best-performing models for fraud detection in this dataset.

---

## **6. Voting Classifier**

- **F1-Score:** 0.4554
- **Precision-Recall AUC (PR-AUC):** 0.8166
- **Confusion Matrix:**
  - True Negative: 56,709
  - False Positive: 161
  - False Negative: 16
  - True Positive: 74
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.31 (Fraud)
  - Recall: 1.00 (Not Fraud), 0.82 (Fraud)
  - Accuracy: 100%
- **Best Threshold:** 0.71
- **Best F1-Score:** 0.8434

**Observation:**
The Voting Classifier offers a strong recall for fraud detection but at the cost of precision. The F1-Score is moderate, indicating some trade-offs between the model's ability to identify frauds and minimizing false positives.

---

### **Summary and Recommendations**

- **Best Overall Model:** Random Forest and the Voting Classifier are the top performers, with high F1-Scores and Precision-Recall AUC. Both models effectively balance precision and recall, making them suitable for fraud detection tasks.
  
- **High Recall Models:** Neural Network and Gradient Boosting are notable for their high recall, essential for minimizing false negatives in fraud detection.

- **Precision vs. Recall Trade-off:** Models like AdaBoost and Logistic Regression emphasize recall but at the cost of low precision, leading to a higher number of false positives.

- **Threshold Tuning:** Adjusting decision thresholds has a significant impact on model performance, especially in imbalanced datasets. Random Forest and Gradient Boosting particularly benefit from threshold optimization.
