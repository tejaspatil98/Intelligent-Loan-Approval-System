# 🏦 SecureTrust Bank - Intelligent Loan Approval System

An end-to-end Supervised Machine Learning pipeline and web application designed to automate loan eligibility evaluations for **SecureTrust Bank**, replacing slow, biased manual reviews with precise data-driven predictions.

---

## 📌 Problem Statement

SecureTrust Bank processes hundreds of loan applications daily across urban and rural India. Manual verification leads to two major problems:
1. **False Rejections:** Good customers are turned away, leading to lost revenue.
2. **False Approvals:** High-risk applicants are approved, causing non-performing assets (NPAs).

**Objective:** Build an automated binary classification system to accurately predict loan approval (`Approved` vs. `Rejected`).

---

## 🛠️ Tech Stack & Tools

* **Language:** Python 3.10+
* **Data Processing & EDA:** Pandas, NumPy, Seaborn, Matplotlib
* **Machine Learning Pipeline:** Scikit-Learn
* **Model Deployment:** Streamlit, Joblib

---

## 🔬 Model Performance & Comparison

Three classifiers namely Logistic regression, KNN and Naive Bayes were trained, feature engineered and evaluated on scaled data (`StandardScaler`). **Gaussian Naive Bayes** achieved the highest overall performance due to its robust handling of probabilistic continuous features.

### Naive Bayes Performance Evaluation

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **86.00%** |
| **Precision** | **81.13%** |
| **Recall** | **70.49%** |
| **F1 Score** | **75.43%** |

### Confusion Matrix
```text
               Predicted No    Predicted Yes
Actual No          129              10
Actual Yes          18              43
```
* **True Negatives (TN):** 129 correctly rejected high-risk profiles.
* **False Positives (FP):** 10 low-risk rejections (Minimized customer loss).

---

## 📁 Repository Structure

```text
├── loan_aprooval_data.csv
├── naive_bayes_model.pkl
├── scaler.pkl
├── credit_wise.ipynb
├── app.py
├── requirements.txt
├── test_model.py
└── README.md
```

---
