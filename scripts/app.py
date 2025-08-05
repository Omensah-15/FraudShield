import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
import json
import os
from sklearn.preprocessing import StandardScaler
from joblib import load
from datetime import datetime, timedelta
import shutil

# Configure logging
logging.basicConfig(
    filename='fraud_detection_predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Directory and retention settings
MODEL_DIR = os.getenv('MODEL_DIR', 'model_artifacts')
PREDICTIONS_DIR = os.getenv('PREDICTIONS_DIR', 'predictions')
RETENTION_DAYS = 7  # Delete predictions/logs older than 7 days

# Streamlit page configuration
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🛡️")

def cleanup_old_files(directory, retention_days=RETENTION_DAYS):
    """Delete files older than retention_days in the specified directory."""
    try:
        now = datetime.now()
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if now - file_mtime > timedelta(days=retention_days):
                    os.remove(file_path)
                    logging.info(f"Deleted old file: {file_path}")
        # Clean up logs
        log_file = 'fraud_detection_predict.log'
        if os.path.exists(log_file):
            log_mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            if now - log_mtime > timedelta(days=retention_days):
                with open(log_file, 'w'):  # Truncate log file
                    pass
                logging.info("Truncated old log file.")
    except Exception as e:
        logging.error(f"Failed to clean up old files: {str(e)}")
        st.warning(f"Could not clean up old files: {str(e)}")

def load_data(file=None):
    """Load transaction data from uploaded file or create a sample dataset."""
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                st.error("Unsupported file format. Please upload a CSV, JSON, or Excel file.")
                logging.error(f"Unsupported file format: {file.name}")
                return None
            logging.info(f"Dataset loaded successfully from {file.name}. Shape: {df.shape}")
            st.success(f"Loaded {file.name} with {df.shape[0]} transactions.")
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            logging.error(f"Error loading file {file.name}: {str(e)}")
            return None
    
    # Create sample dataset
    logging.info("Creating sample dataset.")
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'Transaction_ID': [f'T{i}' for i in range(n_samples)],
        'User_ID': np.random.randint(1000, 5000, n_samples),
        'Transaction_Amount': np.random.lognormal(mean=5, sigma=1.5, size=n_samples).clip(10, 10000),
        'Transaction_Type': np.random.choice(['ATM Withdrawal', 'Bill Payment', 'POS Payment'], n_samples, p=[0.3, 0.3, 0.4]),
        'Time_of_Transaction': np.random.choice(list(range(24)) + [np.nan], n_samples, p=[0.04167]*24 + [0.04]),
        'Device_Used': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples, p=[0.6, 0.3, 0.1]),
        'Location': np.random.choice(['San Francisco', 'New York', 'Chicago', 'Boston', np.nan], n_samples, p=[0.25, 0.25, 0.2, 0.2, 0.1]),
        'Previous_Fraudulent_Transactions': np.random.choice([0, 1, 2, 3], n_samples, p=[0.9, 0.05, 0.03, 0.02]),
        'Account_Age': np.random.randint(1, 120, n_samples),
        'Number_of_Transactions_Last_24H': np.random.randint(1, 20, n_samples),
        'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'UPI', np.nan], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    })
    logging.info("Sample dataset created with realistic distributions.")
    st.info("No file uploaded. Using sample dataset with 100 transactions.")
    return df

def preprocess_data(df, encoders, feature_names, scaler, precomputed_means, precomputed_modes):
    """Prepare data for prediction by cleaning, scaling, and encoding."""
    logging.info("Starting data preprocessing...")
    
    try:
        # Handle missing values
        df = df.fillna(precomputed_means)
        df = df.fillna(precomputed_modes)
        logging.info("Missing values filled with precomputed means and modes.")
        
        # Encode categorical variables
        for col in encoders:
            if col in df.columns:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError:
                    unknown_count = sum(~df[col].astype(str).isin(encoders[col].classes_))
                    logging.warning(f"{unknown_count} unknown categories in {col}. Using default encoding.")
                    df[col] = df[col].astype(str).map(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
                    df[col] = encoders[col].transform(df[col])
        
        # Scale numerical features
        numerical_cols = [col for col in feature_names if df[col].dtype in ['int64', 'float64']]
        if numerical_cols:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
            logging.info(f"Scaled numerical columns: {numerical_cols}")
        
        # Add missing features
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
                logging.warning(f"Added missing feature {col} with default value 0.")
        
        # Keep Transaction_ID, drop for prediction
        if 'Transaction_ID' in df.columns:
            transaction_ids = df['Transaction_ID']
            df = df.drop('Transaction_ID', axis=1)
        else:
            transaction_ids = [f'T{i}' for i in range(len(df))]
        
        # Ensure only expected features
        df = df[feature_names]
        
        logging.info("Data preprocessing completed.")
        return df, transaction_ids
    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        st.error(f"Preprocessing failed: {str(e)}")
        raise

def predict_fraud(df):
    """Predict fraud for transactions in the provided DataFrame."""
    try:
        # Load model and artifacts
        required_files = [
            'ensemble_model.joblib', 'feature_names.joblib', 'scaler.joblib',
            'precomputed_means.joblib', 'precomputed_modes.joblib',
            'encoder_Transaction_Type.joblib', 'encoder_Device_Used.joblib',
            'encoder_Location.joblib', 'encoder_Payment_Method.joblib'
        ]
        for file_name in required_files:
            if not os.path.exists(os.path.join(MODEL_DIR, file_name)):
                raise FileNotFoundError(f"Required file {file_name} not found in {MODEL_DIR}.")
        
        model = load(os.path.join(MODEL_DIR, 'ensemble_model.joblib'))
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
        
        # Load model metadata
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
        metadata = {'Model': 'Unknown', 'Version': 'Unknown', 'Training_Date': 'Unknown'}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Preprocess data
        X, transaction_ids = preprocess_data(df, encoders, feature_names, scaler, precomputed_means, precomputed_modes)
        
        # Make predictions
        predictions = model.predict(X)
        fraud_probabilities = model.predict_proba(X)[:, 1]
        logging.info(f"Predicted {sum(predictions)} fraudulent transactions out of {len(predictions)}.")
        
        # Create results
        results = pd.DataFrame({
            'Transaction_ID': transaction_ids,
            'Fraud_Prediction': ['Fraudulent' if pred == 1 else 'Legitimate' for pred in predictions],
            'Fraud_Probability': fraud_probabilities
        })
        results['Fraud_Probability'] = results['Fraud_Probability'].round(4)
        
        # Save results
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        json_path = os.path.join(PREDICTIONS_DIR, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        excel_path = os.path.join(PREDICTIONS_DIR, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        results.to_json(json_path, orient='records', indent=4)
        results.to_excel(excel_path, index=False)
        
        return results, json_path, excel_path, metadata
    except FileNotFoundError as e:
        logging.error(f"Model or artifact files not found: {str(e)}")
        st.error(f"Error: {str(e)}. Run Fraud_Detection.ipynb to train the model and save artifacts.")
        return None, None, None, None
    except Exception as e:
        logging.error(f"Fraud prediction failed: {str(e)}")
        st.error(f"Prediction failed: {str(e)}")
        return None, None, None, None

def main():
    """Main function for Streamlit app."""
    st.title("🛡️ Fraud Detection Dashboard")
    st.markdown("""
        Upload a transaction dataset (CSV, JSON, or Excel) or use the sample dataset to predict fraudulent transactions.
        Explore predictions with interactive filters, visualizations, and top risky transactions.
    """)

    # Clean up old files
    cleanup_old_files(PREDICTIONS_DIR)
    
    # Sidebar for file upload and filters
    st.sidebar.header("Data Input & Filters")
    uploaded_file = st.sidebar.file_uploader("Upload Transactions", type=['csv', 'json', 'xlsx', 'xls'])
    
    # Load data
    df = load_data(uploaded_file)
    if df is None:
        return
    
    # Predict fraud
    results, json_path, excel_path, metadata = predict_fraud(df)
    if results is None:
        return
    
    # Model metadata
    st.sidebar.header("Model Metadata")
    st.sidebar.write(f"**Model**: {metadata.get('Model', 'Unknown')}")
    st.sidebar.write(f"**Version**: {metadata.get('Version', 'Unknown')}")
    st.sidebar.write(f"**Training Date**: {metadata.get('Training_Date', 'Unknown')}")
    
    # Display summary
    st.header("Prediction Summary")
    fraud_count = len(results[results['Fraud_Prediction'] == 'Fraudulent'])
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(results))
    col2.metric("Fraudulent Transactions", f"{fraud_count} ({fraud_count/len(results)*100:.2f}%)")
    col3.metric("Average Fraud Probability", f"{results['Fraud_Probability'].mean():.4f}")
    
    # Interactive filters
    st.sidebar.subheader("Filter Results")
    prediction_filter = st.sidebar.selectbox("Filter by Prediction", ["All", "Fraudulent", "Legitimate"])
    probability_threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)
    
    filtered_results = results.copy()
    if prediction_filter != "All":
        filtered_results = filtered_results[filtered_results['Fraud_Prediction'] == prediction_filter]
    filtered_results = filtered_results[filtered_results['Fraud_Probability'] >= probability_threshold]
    
    # Top risky transactions
    st.header("Top 5 Risky Transactions")
    top_risky = results.sort_values(by='Fraud_Probability', ascending=False).head(5)
    st.dataframe(top_risky, use_container_width=True)
    
    # Pie chart
    st.header("Fraud Prediction Breakdown")
    pie_data = results['Fraud_Prediction'].value_counts().reset_index()
    pie_data.columns = ['Fraud_Prediction', 'Count']
    fig_pie = px.pie(
        pie_data,
        names='Fraud_Prediction',
        values='Count',
        color='Fraud_Prediction',
        color_discrete_map={'Fraudulent': '#FF4B4B', 'Legitimate': '#4CAF50'},
        title="Breakdown of Fraudulent vs. Legitimate Transactions",
        height=400
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Bar chart visualization
    st.header("Fraud Probability Distribution")
    fig_bar = px.bar(
        filtered_results,
        x='Transaction_ID',
        y='Fraud_Probability',
        color='Fraud_Prediction',
        color_discrete_map={'Fraudulent': '#FF4B4B', 'Legitimate': '#4CAF50'},
        title="Fraud Probabilities by Transaction",
        labels={'Fraud_Probability': 'Probability of Fraud', 'Transaction_ID': 'Transaction ID'},
        height=500
    )
    fig_bar.update_layout(xaxis={'tickangle': 45}, showlegend=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Filtered results
    st.header("Filtered Predictions")
    st.dataframe(filtered_results, use_container_width=True)
    
    # Download buttons
    st.header("Download Results & Logs")
    col1, col2 = st.columns(2)
    with col1:
        with open(json_path, 'r') as f:
            st.download_button(
                label="Download Predictions (JSON)",
                data=f,
                file_name=os.path.basename(json_path),
                mime="application/json"
            )
        with open(excel_path, 'rb') as f:
            st.download_button(
                label="Download Predictions (Excel)",
                data=f,
                file_name=os.path.basename(excel_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    with col2:
        with open('fraud_detection_predict.log', 'r') as f:
            st.download_button(
                label="Download Logs",
                data=f,
                file_name=f"fraud_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
