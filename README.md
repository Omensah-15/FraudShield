# FraudShield: Fraud Detection Pipeline

**FraudShield** is a machine learning-powered fraud detection system designed to identify fraudulent transactions with high accuracy. It includes a complete pipeline for data preprocessing, model training, prediction, and an interactive dashboard for real-time insights.

## Project Overview

FraudShield detects fraudulent activities in transaction datasets using a RandomForestClassifier enhanced with SMOTE (Synthetic Minority Oversampling Technique) for class imbalance. The system features robust feature engineering, logging, and real-time visualization.

### Key Features

- Data Preprocessing: Handles missing values and engineers features like `Transaction_Frequency`, `Amount_ZScore`, and `Is_Night_Transaction`.
- Model Training: Trains a RandomForestClassifier with hyperparameter tuning and SMOTE.
- Interactive Dashboard: Streamlit app for uploading files, visualizing predictions, and downloading results.
- Prediction Logging: Saves prediction outputs in both JSON and Excel formats, including relevant metadata.

## Prerequisites

Ensure the following before running the project:

- Python: Version 3.8 or higher
- Dependencies: Listed in `requirements.txt`

## Installation

1. Set Up the Project Directory
   - Place the project in a directory, e.g.:
     ```
     C:\YourPath\FraudShield
     ```

2. Install Dependencies
   ```bash
   cd C:\YourPath\FraudShield
   pip install -r requirements.txt
````

(Optional: create and activate a virtual environment)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

3. Prepare the Dataset

   * Save your CSV dataset to the `data/` directory:

     ```
     data/Fraud Detection Dataset.csv
     ```

   * Expected columns:

     ```
     Transaction_ID, User_ID, Transaction_Amount, Transaction_Type, Time_of_Transaction, 
     Device_Used, Location, Previous_Fraudulent_Transactions, Account_Age, 
     Number_of_Transactions_Last_24H, Payment_Method, Fraudulent
     ```

Usage

 1. Train the Model

* Launch Jupyter Notebook:

  ```bash
  jupyter notebook
  ```
* Open and run `notebooks/FraudShield.ipynb` to train the model and save artifacts in `model_artifacts/`.

2. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

* A browser will open at `http://localhost:8501`.
* Upload a transaction dataset to get fraud predictions.
* Use filters to view results and download outputs.

3. Run Predictions via Script

```bash
python predict.py
```

* This generates output files in `predictions/`.

Project Structure

```
FraudShield/
├── data/                  # Transaction datasets
├── model_artifacts/       # Trained model and preprocessors
├── predictions/           # Prediction outputs (JSON, Excel)
├── notebooks/             # Jupyter notebooks for experimentation
├── app.py                 # Streamlit dashboard
├── predict.py             # Standalone prediction script
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

Customization

* Paths: Update paths in `app.py`, `predict.py`, or the notebook to match your file system.
* Feature Engineering: Modify engineered features in the notebook or script if your dataset changes.
* Model Tuning: Adjust the parameter grid in the notebook for better performance.

Contributing

Contributions are welcome! You can:

* Fork the repository and open a pull request
* Suggest improvements via Issues or Discussions

License

Currently not under a formal license. Intended for educational and non-commercial use. For other purposes, please reach out.

Acknowledgments

* Built using Scikit-learn, Pandas, Streamlit, and Imbalanced-learn.
```
👨‍💻 Author

**Obed Mensah**  
*Data Scientist — Python | Power BI | SQL | Analytics*  
📧 [heavenzlebron7@gmail.com](mailto:heavenzlebron7@gmail.com)

