# fraudshield
import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 1. Embed Training Data
# =======================
TRAINING_DATA = """
TransactionID,Amount,Transaction_Type,CustomerID,MerchantID,IsFraud
T1,120,Online,C1,M1,0
T2,500,POS,C2,M1,1
T3,70,Online,C1,M2,0
T4,2000,POS,C3,M3,1
T5,130,Online,C4,M2,0
T6,220,POS,C2,M1,0
T7,90,Online,C5,M3,0
T8,1600,POS,C6,M2,1
T9,45,Online,C7,M2,0
T10,400,POS,C1,M1,1
"""

# ==========================
# 2. Preprocessing + Training
# ==========================
@st.cache_resource
def load_model():
    # Load training data
    df = pd.read_csv(io.StringIO(TRAINING_DATA.strip()))

    # Encode categorical variables
    encoders = {}
    for col in ["Transaction_Type", "CustomerID", "MerchantID"]:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

    # Feature engineering
    df["Transaction_Frequency"] = df.groupby("CustomerID")["TransactionID"].transform("count")
    df["Amount_ZScore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["Amount_Anomaly"] = (df["Amount_ZScore"].abs() > 3).astype(int)
    df["Freq_Anomaly"] = (df["Transaction_Frequency"] > df["Transaction_Frequency"].mean() + 2*df["Transaction_Frequency"].std()).astype(int)

    X = df.drop(columns=["TransactionID", "IsFraud"])
    y = df["IsFraud"]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Train RandomForest with RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    clf = RandomizedSearchCV(rf, param_distributions=param_grid,
                             n_iter=5, cv=3, scoring='roc_auc',
                             random_state=42, n_jobs=-1)
    clf.fit(X_resampled, y_resampled)

    return clf.best_estimator_, scaler, encoders, list(X.columns)

model, scaler, encoders, feature_names = load_model()

# ==========================
# 3. Preprocess New Data
# ==========================
def preprocess_data(df):
    df = df.copy()

    # Handle missing values
    df.fillna(0, inplace=True)

    # Encode with fitted encoders
    for col in ["Transaction_Type", "CustomerID", "MerchantID"]:
        if col in df.columns:
            df[col] = df[col].map(lambda s: encoders[col].transform([s])[0] if s in encoders[col].classes_ else -1)

    # Feature engineering
    if "TransactionID" not in df.columns:
        df["TransactionID"] = [f"TX{i}" for i in range(len(df))]

    if "CustomerID" in df.columns:
        df["Transaction_Frequency"] = df.groupby("CustomerID")["TransactionID"].transform("count")
    else:
        df["Transaction_Frequency"] = 1

    df["Amount_ZScore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-6)
    df["Amount_Anomaly"] = (df["Amount_ZScore"].abs() > 3).astype(int)
    df["Freq_Anomaly"] = (df["Transaction_Frequency"] > df["Transaction_Frequency"].mean() + 2*df["Transaction_Frequency"].std()).astype(int)

    # Align features
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale
    X_scaled = scaler.transform(df)
    return X_scaled, df

# ==========================
# 4. Streamlit UI
# ==========================
st.set_page_config(page_title="FraudShield", layout="wide")
st.title("🛡️ FraudShield - Fraud Detection Dashboard")

st.sidebar.header("Upload Transaction Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV, JSON, or Excel", type=["csv", "json", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        data = pd.read_json(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    X_scaled, processed_df = preprocess_data(data)

    # Predictions
    fraud_probs = model.predict_proba(X_scaled)[:, 1]
    fraud_preds = (fraud_probs > 0.5).astype(int)

    processed_df["Fraud_Probability"] = fraud_probs
    processed_df["Predicted_Fraud"] = fraud_preds

    st.subheader("Fraud Predictions")
    st.dataframe(processed_df.head())

    # Stats
    fraud_counts = processed_df["Predicted_Fraud"].value_counts()
    st.subheader("Fraud Stats")
    st.bar_chart(fraud_counts)

    # ROC Curve
    fpr, tpr, _ = roc_curve(processed_df["Predicted_Fraud"], processed_df["Fraud_Probability"])
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc_score(processed_df['Predicted_Fraud'], processed_df['Fraud_Probability']):.2f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)

    # Download results
    st.subheader("Download Predictions")
    csv = processed_df.to_csv(index=False).encode()
    st.download_button("Download as CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

else:
    st.info("Please upload a dataset to start fraud detection.")
