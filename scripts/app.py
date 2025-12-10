import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import logging
import json
import os
from datetime import datetime, timedelta
from joblib import load

# ---------------------- CONFIG ----------------------
logging.basicConfig(
    filename='fraud_detection_predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_DIR = os.getenv('MODEL_DIR', 'model_artifacts')
PREDICTIONS_DIR = os.getenv('PREDICTIONS_DIR', 'predictions')
RETENTION_DAYS = 7

st.set_page_config(page_title='FraudShield Dashboard', layout='wide', page_icon='🛡️')

# ---------------------- CLEANUP ----------------------
def cleanup_old_files(directory, retention_days=RETENTION_DAYS):
    try:
        now = datetime.now()
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if now - file_mtime > timedelta(days=retention_days):
                    os.remove(file_path)
        log_file = 'fraud_detection_predict.log'
        if os.path.exists(log_file):
            log_mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            if now - log_mtime > timedelta(days=retention_days):
                with open(log_file, 'w'): pass
    except Exception as e:
        logging.error(f'Cleanup failed: {e}')
        st.warning(f'Cleanup failed: {e}')

# ---------------------- CACHED MODEL LOADING ----------------------
@st.cache_resource
def load_model():
    model = load(os.path.join(MODEL_DIR, 'ensemble_model.joblib.gz'))
    feature_names = load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
    scaler = load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    precomputed_means = load(os.path.join(MODEL_DIR, 'precomputed_means.joblib'))
    precomputed_modes = load(os.path.join(MODEL_DIR, 'precomputed_modes.joblib'))
    encoders = {
        'Transaction_Type': load(os.path.join(MODEL_DIR, 'encoder_Transaction_Type.joblib')),
        'Device_Used': load(os.path.join(MODEL_DIR, 'encoder_Device_Used.joblib')),
        'Location': load(os.path.join(MODEL_DIR, 'encoder_Location.joblib')),
        'Payment_Method': load(os.path.join(MODEL_DIR, 'encoder_Payment_Method.joblib'))
    }
    return model, feature_names, scaler, precomputed_means, precomputed_modes, encoders

# ---------------------- DATA LOADING ----------------------
@st.cache_data
def load_data(file=None):
    import chardet
    if file is not None:
        try:
            if isinstance(file, str):
                with open(file, "rb") as f:
                    encoding = chardet.detect(f.read())['encoding']
                if file.endswith('.csv'):
                    df = pd.read_csv(file, encoding=encoding)
                elif file.endswith('.json'):
                    df = pd.read_json(file)
                elif file.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                else:
                    raise ValueError('Unsupported format')
            else:
                if file.name.endswith('.csv'):
                    encoding = chardet.detect(file.read())['encoding']
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                elif file.name.endswith('.json'):
                    df = pd.read_json(file)
                elif file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                else:
                    raise ValueError('Unsupported format')
            return df
        except Exception as e:
            st.error(f'Error loading file: {e}')
            return None
    # Default small sample
    return pd.read_csv('data/sample_transactions.csv')

# ---------------------- DATA PREPROCESSING ----------------------
@st.cache_data
def preprocess_data(df, encoders, feature_names, scaler, precomputed_means, precomputed_modes):
    try:
        df['Transaction_Frequency'] = df.groupby('User_ID')['Transaction_ID'].transform('count') / df['Account_Age'].replace('', 1).astype(float)
        df['Amount_ZScore'] = (df['Transaction_Amount'] - df.groupby('User_ID')['Transaction_Amount'].transform('mean')) / df.groupby('User_ID')['Transaction_Amount'].transform('std').fillna(1)
        df['Is_Night_Transaction'] = df['Time_of_Transaction'].replace('', 0).astype(float).apply(lambda x: 1 if 0 <= x <= 6 else 0)
        df['Transaction_Velocity'] = df['Number_of_Transactions_Last_24H'] / (df['Account_Age'].replace('', 1).astype(float)/30).clip(lower=1)
        df['Location_Anomaly'] = df.groupby('User_ID')['Location'].transform(lambda x: 1 if x.nunique()>2 else 0)
        df['Transaction_Acceleration'] = df['Number_of_Transactions_Last_24H'] / df.groupby('User_ID')['Number_of_Transactions_Last_24H'].transform('mean').clip(lower=1)
        df['Device_Anomaly'] = df.groupby('User_ID')['Device_Used'].transform(lambda x: 1 if x.nunique()>2 else 0)

        df = df.fillna({
            'Transaction_Type':'Unknown', 'Device_Used':'Unknown',
            'Location':'Unknown', 'Payment_Method':'Unknown',
            'Time_of_Transaction':0, 'Account_Age':1
        })
        df = df.fillna(precomputed_means).fillna(precomputed_modes)

        for col in encoders:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
                df[col] = encoders[col].transform(df[col])

        numerical_cols = [c for c in feature_names if c in df.columns and df[c].dtype in ['int64','float64']]
        if numerical_cols: df[numerical_cols] = scaler.transform(df[numerical_cols])

        for col in feature_names:
            if col not in df.columns: df[col]=0

        transaction_ids = df['Transaction_ID'] if 'Transaction_ID' in df.columns else [f'T{i}' for i in range(len(df))]
        df = df[feature_names]

        return df, transaction_ids
    except Exception as e:
        st.error(f'Preprocessing failed: {e}')
        return None, None

# ---------------------- PREDICTION ----------------------
def predict_fraud(df):
    model, feature_names, scaler, precomputed_means, precomputed_modes, encoders = load_model()
    X, transaction_ids = preprocess_data(df, encoders, feature_names, scaler, precomputed_means, precomputed_modes)
    if X is None: return None, None
    predictions = model.predict(X)
    prob = model.predict_proba(X)[:,1]
    results = pd.DataFrame({
        'Transaction_ID': transaction_ids,
        'Fraud_Prediction': ['Fraudulent' if p==1 else 'Legitimate' for p in predictions],
        'Fraud_Probability': prob.round(4)
    })
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results.to_json(f'{PREDICTIONS_DIR}/predictions_{timestamp}.json', orient='records', indent=4)
    results.to_excel(f'{PREDICTIONS_DIR}/predictions_{timestamp}.xlsx', index=False)
    return results

# ---------------------- DASHBOARD ----------------------
def main():
    st.title('🛡️ FraudShield Dashboard')
    st.markdown("Upload transactions to detect fraud. Professional, fast, and interactive.")

    cleanup_old_files(PREDICTIONS_DIR)

    uploaded_file = st.sidebar.file_uploader('Upload Transactions', type=['csv','json','xlsx','xls'])
    df = load_data(uploaded_file)
    if df is None: return

    with st.spinner('Predicting fraud...'):
        results = predict_fraud(df)
    if results is None: return

    # Metrics
    fraud_count = results.Fraud_Prediction.value_counts().get('Fraudulent',0)
    total = len(results)
    col1,col2,col3 = st.columns(3)
    col1.metric('Total Transactions', total)
    col2.metric('Fraudulent Transactions', f'{fraud_count} ({fraud_count/total*100:.2f}%)')
    col3.metric('Average Fraud Probability', f'{results["Fraud_Probability"].mean():.4f}')

    # Top Risky
    st.subheader('Top 5 Risky Transactions')
    st.dataframe(results.sort_values('Fraud_Probability', ascending=False).head(5), use_container_width=True)

    # Visualizations
    st.subheader('Fraud Distribution by User')
    fig1 = px.histogram(results, x='Fraud_Probability', color='Fraud_Prediction', nbins=20, template='plotly_dark')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader('Fraud Count by Prediction')
    fig2 = px.pie(results, names='Fraud_Prediction', values='Fraud_Probability', color='Fraud_Prediction', template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

    # Download buttons
    st.subheader('Download Results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.download_button('Download Excel', data=open(f'{PREDICTIONS_DIR}/predictions_{timestamp}.xlsx','rb').read(),
                       file_name=f'predictions_{timestamp}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.download_button('Download JSON', data=open(f'{PREDICTIONS_DIR}/predictions_{timestamp}.json','r').read(),
                       file_name=f'predictions_{timestamp}.json', mime='application/json')

if __name__ == '__main__':
    main()



