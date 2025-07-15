import pandas as pd
import numpy as np
import logging
import json
import os
from sklearn.preprocessing import StandardScaler
from joblib import load

# Configure logging to track what the script does
logging.basicConfig(
    filename='fraud_detection_predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path=None):
    """Load the transaction data CSV or create a sample if file_path is invalid."""
    if file_path is None:
        file_path = input("Enter the path to transaction data CSV")
    
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.warning(f"File {file_path} not found. Creating sample dataset.")
        print(f"File {file_path} not found. Using sample dataset.")
        np.random.seed(42)
        n_samples = 10
        df = pd.DataFrame({
            'Transaction_ID': [f'T{i}' for i in range(n_samples)],
            'User_ID': np.random.randint(1000, 5000, n_samples),
            'Transaction_Amount': np.random.lognormal(mean=5, sigma=1, size=n_samples),
            'Transaction_Type': np.random.choice(['ATM Withdrawal', 'Bill Payment', 'POS Payment'], n_samples),
            'Time_of_Transaction': np.random.choice(list(range(24)) + [np.nan], n_samples, p=[0.04167]*24 + [0.04]),
            'Device_Used': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples),
            'Location': np.random.choice(['San Francisco', 'New York', 'Chicago', 'Boston', np.nan], n_samples),
            'Previous_Fraudulent_Transactions': np.random.randint(0, 5, n_samples),
            'Account_Age': np.random.randint(1, 120, n_samples),
            'Number_of_Transactions_Last_24H': np.random.randint(1, 15, n_samples),
            'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'UPI', np.nan], n_samples)
        })
        logging.info("Sample dataset created.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df, encoders, feature_names):
    """Prepare the data for prediction by cleaning and encoding it."""
    logging.info("Starting data preprocessing...")
    
    try:
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))  # Fill numerical NaNs with mean
        df = df.fillna(df.mode().iloc[0])  # Fill categorical NaNs with mode
        
        # Encode categorical variables (e.g., 'Credit Card' to 0)
        for col in encoders:
            if col in df.columns:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError:
                    logging.warning(f"Unknown category in {col}. Using default encoding.")
                    df[col] = encoders[col].transform([encoders[col].classes_[0]])[0]
        
        # Add missing features with default value (0)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Keep Transaction_ID for output, but drop it for prediction
        if 'Transaction_ID' in df.columns:
            transaction_ids = df['Transaction_ID']
            df = df.drop('Transaction_ID', axis=1)
        else:
            transaction_ids = [f'T{i}' for i in range(len(df))]
        
        # Ensure only expected features are used
        df = df[feature_names]
        
        logging.info("Data preprocessing completed.")
        return df, transaction_ids
    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise

def predict_fraud(file_path=None):
    """Predict fraud for transactions in the provided CSV."""
    try:
        # Load the trained model and encoders
        model = load('model_artifacts/ensemble_model.joblib')
        feature_names = load('model_artifacts/feature_names.joblib')
        encoders = {
            'Transaction_Type': load('encoder_Transaction_Type.joblib'),
            'Device_Used': load('encoder_Device_Used.joblib'),
            'Location': load('encoder_Location.joblib'),
            'Payment_Method': load('encoder_Payment_Method.joblib')
        }
        
        # Load new transaction data
        df = load_data(file_path)
        
        # Preprocess data
        X, transaction_ids = preprocess_data(df, encoders, feature_names)
        
        # Make predictions
        predictions = model.predict(X)
        fraud_probabilities = model.predict_proba(X)[:, 1]
        
        # Create results
        results = [
            {
                'Transaction_ID': tid,
                'Fraud_Prediction': 'Fraudulent' if pred == 1 else 'Legitimate',
                'Fraud_Probability': float(prob)
            } for tid, pred, prob in zip(transaction_ids, predictions, fraud_probabilities)
        ]
        
        # Save results to JSON and Excel
        os.makedirs('predictions', exist_ok=True)
        with open('predictions/predictions.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        results_df = pd.DataFrame(results)
        results_df.to_excel('predictions/predictions.xlsx', index=False)
        
        print("\nFraud Detection Results:")
        print(results_df)
        print("\nResults saved to predictions/predictions.json and predictions/predictions.xlsx")
        
        logging.info("Fraud prediction completed.")
        return results_df
    except FileNotFoundError:
        logging.error("Model or encoder files not found. Please run Fraud_Detection.ipynb first.")
        print("Error: Model or encoder files not found. Run Fraud_Detection.ipynb to train the model.")
    except Exception as e:
        logging.error(f"Fraud prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    predict_fraud()