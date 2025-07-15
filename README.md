# 🕵️‍♂️ Fraud Detection Pipeline using Machine Learning
---
## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](https://github.com/Omensah-15/fraud-detection-pipeline/blob/main/README.md#-dataset-overview)
- [Notebook Walkthrough](https://github.com/Omensah-15/fraud-detection-pipeline/blob/main/README.md#-project-goals)
- [Exploratory Data Analysis (EDA)](https://github.com/Omensah-15/fraud-detection-pipeline/blob/main/README.md#%EF%B8%8F-step-3-exploratory-data-analysis-eda)
- [Model Training & Evaluation](https://github.com/Omensah-15/fraud-detection-pipeline/tree/main#-model-training--evaluation)
- [Prediction Script](https://github.com/Omensah-15/fraud-detection-pipeline/tree/main#-standalone-prediction-script)
---
## 📌 Project Overview

An end-to-end machine learning pipeline to detect fraudulent transactions in financial data.  
This project is built to address real-world challenges such as:

- ⚖️ Imbalanced class distribution  
- 🧮 Categorical variable encoding  
- 📊 Feature scaling and engineering  
- 🔀 Ensemble modeling with Random Forest & XGBoost  

It also includes a **production-ready prediction script** for fraud detection on new transaction data, making it suitable for deployment or batch processing in real-world systems.

---

## 📊 Dataset Overview

This dataset consists of **51,000+ real-world inspired transactions**, each labeled as **fraudulent** or **legitimate**, and includes various transaction and user behavior features.

### 🔍 Features:
- `Transaction_Amount`, `Transaction_Type`, `Payment_Method`
- `Device_Used`, `Location`, `Time_of_Transaction`
- `Previous_Fraudulent_Transactions`, `Account_Age`, `Number_of_Transactions_Last_24H`
- `Fraudulent`: (Target variable — 1 = Fraud, 0 = Legitimate)

---

## 🎯 Project Goals

- Build a scalable fraud detection system using classical ML models.
- Tackle **class imbalance** using SMOTE.
- Achieve **>90% test accuracy** with generalizable models.
- Provide an **automated, reproducible pipeline** for future datasets.

---

## 🛠️ Pipeline Summary

### ✔️ Step 1: Setup and Dependency Check
All required packages are verified before execution:


### ✔️ Step 2: Data Preparation
- Training set shape: `(40800, 10)`
- Test set shape: `(10200, 10)`
- After SMOTE resampling: `(77584, 10)`

### ✔️ Step 3: Exploratory Data Analysis (EDA)
Saved plots:
- [`eda_plots/fraud_distribution.png`](eda_plots/fraud_distribution.png)
- [`eda_plots/amount_distribution.png`](eda_plots/amount_distribution.png)
- [`eda_plots/correlation_heatmap.png`](eda_plots/correlation_heatmap.png)

### ✔️ Step 4: Data Preprocessing
- Missing values handled (mean/mode).
- Categorical features encoded with `LabelEncoder`.
- Scaled with `StandardScaler` using pipelines.
- `Transaction_ID` excluded from training.

### ✔️ Step 5: Class Imbalance Handling
Used **SMOTE** to oversample fraudulent cases and balance the training dataset.

---

## 🤖 Model Training & Evaluation

Two models were trained with **GridSearchCV** on hyperparameters and compared via test accuracy, classification report, and ROC curve.

### 🔢 Random Forest
- **Accuracy:** `0.7857`
- **Recall (fraud class):** `0.1932`
- Moderate performance, struggled with minority class.

### ⚡ XGBoost
- **Accuracy:** `0.9294`
- **Recall (fraud class):** `0.0299`
- Much better overall accuracy, but poor recall on fraud.

### 🧠 Ensemble (Voting Classifier)
- **Accuracy:** `0.9130`
- Balanced prediction using RF + XGBoost

### 🏆 Best Model Selected: **XGBoost**
> ✅ **Best model accuracy: 0.9294**  
> 🎯 **Accuracy goal of 90%+ achieved!**

---

## 📦 Artifacts Saved

| File | Description |
|------|-------------|
| `ensemble_model.joblib` | Trained ensemble model |
| `feature_names.joblib` | Feature list for inference |
| `encoder_*.joblib` | Label encoders for categorical variables |
| `processed_transactions.xlsx` | Cleaned and prepared dataset |

---

## 📤 Standalone Prediction Script

Use `predict_fraud.py` to run predictions on new transaction data.

### 💻 Run Command:
```bash
python scripts/predict_fraud.py


## 📄 Sample Output Format

| Transaction_ID | Fraud_Prediction | Fraud_Probability |
|----------------|------------------|-------------------|
| T0001          | Legitimate       | 0.06              |
| T0002          | Fraudulent       | 0.93              |
| T0003          | Legitimate       | 0.12              |


## 👨‍💻 Author

**Obed Mensah**  
*Data Scientist — Python | Power BI | SQL | Analytics*  
📧 [heavenzlebron7@gmail.com](mailto:heavenzlebron7@gmail.com)
