
# **Credit Card Fraud Detection Model Evaluation Report  (Undersampling )**

## **1. Logistic Regression**

- **F1-Score:** 0.0837
- **Precision-Recall AUC (PR-AUC):** 0.6638
- **Confusion Matrix:**
  - True Negative: 55,106
  - False Positive: 1,764
  - False Negative: 9
  - True Positive: 81
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.04 (Fraud)
  - Recall: 0.97 (Not Fraud), 0.90 (Fraud)
  - Accuracy: 97%
- **Best Threshold:** 0.99
- **F1-Score (Best Threshold):** 0.4702

**Observation:**
Logistic Regression's performance is modest, with low precision and F1-Score for fraud detection. The model struggles with identifying fraud cases accurately despite good recall.

---

## **2. Random Forest**

- **F1-Score:** 0.1309
- **Precision-Recall AUC (PR-AUC):** 0.7818
- **Confusion Matrix:**
  - True Negative: 55,775
  - False Positive: 1,095
  - False Negative: 7
  - True Positive: 83
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.07 (Fraud)
  - Recall: 0.98 (Not Fraud), 0.92 (Fraud)
  - Accuracy: 98%
- **Best Threshold:** 0.88
- **F1-Score (Best Threshold):** 0.8156

**Observation:**
Random Forest shows an improvement over Logistic Regression, with better recall for fraud cases but still faces challenges with precision, leading to a relatively low F1-Score.

---

## **3. Gradient Boosting**

- **F1-Score:** 0.0541
- **Precision-Recall AUC (PR-AUC):** 0.4977
- **Confusion Matrix:**
  - True Negative: 53,905
  - False Positive: 2,965
  - False Negative: 5
  - True Positive: 85
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.03 (Fraud)
  - Recall: 0.95 (Not Fraud), 0.94 (Fraud)
  - Accuracy: 95%
- **Best Threshold:** 0.99
- **F1-Score (Best Threshold):** 0.5948

**Observation:**
Gradient Boosting struggles with precision, resulting in a very low F1-Score. The model is effective at recalling fraud cases but has a high false positive rate.

---

## **4. AdaBoost**

- **F1-Score:** 0.0503
- **Precision-Recall AUC (PR-AUC):** 0.6510
- **Confusion Matrix:**
  - True Negative: 53,746
  - False Positive: 3,124
  - False Negative: 7
  - True Positive: 83
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.03 (Fraud)
  - Recall: 0.95 (Not Fraud), 0.92 (Fraud)
  - Accuracy: 95%
- **Best Threshold:** 0.68
- **F1-Score (Best Threshold):** 0.6897

**Observation:**
AdaBoost also exhibits a high number of false positives and a low F1-Score. The model's ability to detect fraud cases is moderate, but it requires significant improvement in precision.

---

## **5. Neural Network**

- **F1-Score:** 0.1203
- **Precision-Recall AUC (PR-AUC):** 0.7159
- **Confusion Matrix:**
  - True Negative: 55,710
  - False Positive: 1,160
  - False Negative: 10
  - True Positive: 80
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.06 (Fraud)
  - Recall: 0.98 (Not Fraud), 0.89 (Fraud)
  - Accuracy: 98%
- **Best Threshold:** 0.95
- **F1-Score (Best Threshold):** 0.8235

**Observation:**
The Neural Network model shows a balance between precision and recall with a higher F1-Score when the threshold is optimized. It is effective at fraud detection with fewer false positives compared to other models.

---

## **6. Voting Classifier**

- **F1-Score:** 0.1119
- **Precision-Recall AUC (PR-AUC):** 0.7401
- **Confusion Matrix:**
  - True Negative: 55,560
  - False Positive: 1,310
  - False Negative: 7
  - True Positive: 83
- **Classification Report:**
  - Precision: 1.00 (Not Fraud), 0.06 (Fraud)
  - Recall: 0.98 (Not Fraud), 0.92 (Fraud)
  - Accuracy: 98%
- **Best Threshold:** 0.94
- **F1-Score (Best Threshold):** 0.8235

**Observation:**
The Voting Classifier performs similarly to the Neural Network, with a good F1-Score and PR-AUC. It balances precision and recall effectively, making it a strong contender for fraud detection.

---

## **Summary and Recommendations**

- **Top Performers:** The Neural Network and Voting Classifier models show the best performance, with high F1-Scores and PR-AUC, effectively balancing precision and recall for fraud detection.

- **Challenges with Other Models:** Logistic Regression, Random Forest, Gradient Boosting, and AdaBoost exhibit lower F1-Scores and high false positive rates, indicating a need for further refinement or alternative approaches.

- **Threshold Optimization:** The importance of threshold tuning is evident, especially for Neural Network and Voting Classifier models, where adjusting the threshold significantly impacts the F1-Score.

- **Overall Insights:** Undersampling has impacted the models' ability to detect fraud cases accurately, often resulting in lower F1-Scores compared to oversampling. It is crucial to carefully select the technique based on the trade-offs between precision and recall.

---
