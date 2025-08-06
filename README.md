# 🛡FraudShield: Fraud Detection Pipeline

**FraudShield** is a machine learning-powered fraud detection system designed to identify fraudulent transactions with high accuracy. It includes a complete pipeline for data preprocessing, model training, prediction, and an interactive dashboard for real-time insights.

## Project Overview

FraudShield detects fraudulent activities in transaction datasets using a RandomForestClassifier enhanced with SMOTE (Synthetic Minority Oversampling Technique) for class imbalance. The system features robust feature engineering, logging, and real-time visualization.

### Key Features

- **Data Preprocessing**: Handles missing values and engineers features like `Transaction_Frequency`, `Amount_ZScore`, and `Is_Night_Transaction`.
- **Model Training**: Trains a RandomForestClassifier with hyperparameter tuning and SMOTE.
- **Interactive Dashboard**: Streamlit app for uploading files, visualizing predictions, and downloading results.
- **Prediction Logging**: Saves prediction outputs in both JSON and Excel formats, including relevant metadata.

## Prerequisites

Ensure the following before running the project:

- **Python**: Version 3.8 or higher
- **Dependencies**: Listed in `requirements.txt`

## Installation

1. **Set Up the Project Directory**
   - Place the project in a directory, e.g.:
     ```
     C:\YourPath\FraudShield
     ```

2. **Install Dependencies**
   ```bash
   cd C:\YourPath\FraudShield
   pip install -r requirements.txt

## 👨‍💻 Author

**Obed Mensah**  
*Data Scientist — Python | Power BI | SQL | Analytics*  
📧 [heavenzlebron7@gmail.com](mailto:heavenzlebron7@gmail.com)

