import logging
from datetime import datetime
from joblib import load
import pandas as pd
import os
import chardet

logging.basicConfig(
    filename='fraud_detection_predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_DIR = os.getenv('MODEL_DIR', 'model_artifacts')
PREDICTIONS_DIR = os.getenv('PREDICTIONS_DIR', 'predictions')

def load_data(file=None):
    if file is not None:
        try:
            if isinstance(file, str):
                with open(file, "rb") as f:
                    result = chardet.detect(f.read())
                    encoding = result['encoding']
                if file.endswith('.csv'):
                    df = pd.read_csv(file, encoding=encoding)
                elif file.endswith('.json'):
                    df = pd.read_json(file)
                elif file.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                else:
                    raise ValueError('Unsupported file format. Use CSV, JSON, or Excel.')
            else:
                if file.name.endswith('.csv'):
                    with open(file.name, "rb") as f:
                        result = chardet.detect(f.read())
                        encoding = result['encoding']
                    df = pd.read_csv(file, encoding=encoding)
                elif file.name.endswith('.json'):
                    df = pd.read_json(file)
                elif file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                else:
                    raise ValueError('Unsupported file format. Use CSV, JSON, or Excel.')
            logging.info(f'Dataset loaded successfully from {file}. Shape: {df.shape}')
            return df
        except Exception as e:
            logging.error(f'Error loading file {file}: {str(e)}')
            raise
    logging.info('Loading sample dataset.')
    df = pd.read_csv('data/sample_transactions.csv')
    return df

def preprocess_data(df, encoders, feature_names, scaler, precomputed_means, precomputed_modes):
    logging.info('Starting data preprocessing...')
    try:
        df['Transaction_Frequency'] = df.groupby('User_ID')['Transaction_ID'].transform('count') / df['Account_Age'].replace('', 1).astype(float)
        df['Amount_ZScore'] = (df['Transaction_Amount'] - df.groupby('User_ID')['Transaction_Amount'].transform('mean')) / df.groupby('User_ID')['Transaction_Amount'].transform('std').fillna(1)
        df['Is_Night_Transaction'] = df['Time_of_Transaction'].replace('', 0).astype(float).apply(lambda x: 1 if 0 <= x <= 6 else 0)
        df['Transaction_Velocity'] = df['Number_of_Transactions_Last_24H'] / (df['Account_Age'].replace('', 1).astype(float) / 30).clip(lower=1)
        df['Location_Anomaly'] = df.groupby('User_ID')['Location'].transform(lambda x: 1 if x.nunique() > 2 else 0)
        df['Transaction_Acceleration'] = df['Number_of_Transactions_Last_24H'] / df.groupby('User_ID')['Number_of_Transactions_Last_24H'].transform('mean').clip(lower=1)
        df['Device_Anomaly'] = df.groupby('User_ID')['Device_Used'].transform(lambda x: 1 if x.nunique() > 2 else 0)

        df = df.fillna({
            'Transaction_Type': 'Unknown',
            'Device_Used': 'Unknown',
            'Location': 'Unknown',
            'Payment_Method': 'Unknown',
            'Time_of_Transaction': 0,
            'Account_Age': 1
        })
        df = df.fillna(precomputed_means).fillna(precomputed_modes)
        logging.info('Missing values filled with precomputed means and modes.')
        
        for col in encoders:
            if col in df.columns:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError:
                    unknown_count = sum(~df[col].astype(str).isin(encoders[col].classes_))
                    logging.warning(f'{unknown_count} unknown categories in {col}. Using default encoding.')
                    df[col] = df[col].astype(str).map(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
                    df[col] = encoders[col].transform(df[col])
        
        numerical_cols = [col for col in feature_names if col in df.columns and df[col].dtype in ['int64', 'float64']]
        if numerical_cols:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
            logging.info(f'Scaled numerical columns: {numerical_cols}')
        
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
                logging.warning(f'Added missing feature {col} with default value 0.')
        
        if 'Transaction_ID' in df.columns:
            transaction_ids = df['Transaction_ID']
            df = df.drop('Transaction_ID', axis=1)
        else:
            transaction_ids = [f'T{i}' for i in range(len(df))]
        
        df = df[feature_names]
        logging.info('Data preprocessing completed.')
        return df, transaction_ids
    except Exception as e:
        logging.error(f'Data preprocessing failed: {str(e)}')
        raise

def predict_fraud(df=None):
    try:
        # Updated to load compressed model file
        required_files = [
            'ensemble_model.joblib.gz', 'feature_names.joblib', 'scaler.joblib',
            'precomputed_means.joblib', 'precomputed_modes.joblib',
            'encoder_Transaction_Type.joblib', 'encoder_Device_Used.joblib',
            'encoder_Location.joblib', 'encoder_Payment_Method.joblib'
        ]
        for file_name in required_files:
            if not os.path.exists(os.path.join(MODEL_DIR, file_name)):
                raise FileNotFoundError(f'Required file {file_name} not found in {MODEL_DIR}. Please upload artifacts or run the training script.')

        # Load compressed model
        model = load(os.path.join(MODEL_DIR, 'ensemble_model.joblib.gz'))  # Handles .gz compression natively
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
        
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
        metadata = {'Model': 'Unknown', 'Version': 'Unknown', 'Training_Date': 'Unknown', 'Accuracy': 0.0}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        if df is None:
            df = load_data()
        
        X, transaction_ids = preprocess_data(df, encoders, feature_names, scaler, precomputed_means, precomputed_modes)
        
        predictions = model.predict(X)
        fraud_probabilities = model.predict_proba(X)[:, 1]
        logging.info(f'Predicted {sum(predictions)} fraudulent transactions out of {len(predictions)}.')
        
        results = pd.DataFrame({
            'Transaction_ID': transaction_ids,
            'Fraud_Prediction': ['Fraudulent' if pred == 1 else 'Legitimate' for pred in predictions],
            'Fraud_Probability': fraud_probabilities
        })
        results['Fraud_Probability'] = results['Fraud_Probability'].round(4)
        
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        json_path = os.path.join(PREDICTIONS_DIR, f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        excel_path = os.path.join(PREDICTIONS_DIR, f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        results.to_json(json_path, orient='records', indent=4)
        results.to_excel(excel_path, index=False)
        
        return results, json_path, excel_path, metadata
    except FileNotFoundError as e:
        logging.error(f'Model or artifact files not found: {str(e)}')
        raise
    except Exception as e:
        logging.error(f'Fraud prediction failed: {str(e)}')
        raise

if __name__ == '__main__':
    # Example usage for local testing or GitHub Actions
    results, json_path, excel_path, metadata = predict_fraud('data/sample_transactions.csv')
    print('Prediction Results:')
    print(results.head())
    print(f'Results saved to {json_path} and {excel_path}')
    print('Metadata:', metadata)
