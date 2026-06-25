# Customer Churn Prediction System with REST API

A complete machine learning pipeline to preprocess telco customer data, train predictive models, and serve predictions via a Flask REST API.

## Overview

Customer churn refers to the rate at which customers stop doing business with an entity. Predicting churn is vital for subscription-based businesses, such as telecommunications companies, because retaining existing customers is significantly more cost-effective than acquiring new ones. This project provides an end-to-end workflow to load and clean raw customer data, perform feature engineering/encoding, train multiple classification models, evaluate their performance, select the best model, and expose a REST API endpoint for real-time inference.

## Dataset

This project uses the **Telco Customer Churn dataset** from Kaggle. The dataset contains 7,043 customer records with 21 features capturing customer demographic details, services subscribed to, account information, and whether the customer churned within the last month.

* **Dataset Path:** `C:\projects\churn_project\data\WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Tech Stack

* **Language:** Python
* **Data Processing & Storage:** Pandas, NumPy, SQLite3
* **Machine Learning:** Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting Classifier), Joblib
* **API Framework:** Flask

## Project Structure

```text
C:\projects\churn_project\
├── data\
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
├── preprocess.py                             # Data loading, cleaning, and DB export
├── train.py                                  # Model training and comparison
├── app.py                                    # Flask REST API
├── test_api.py                               # Sample prediction test script
├── requirements.txt                          # Python package dependencies
├── README.md                                 # Documentation
├── churn.db                                  # Cleaned SQLite database (generated)
├── model.pkl                                 # Best saved model (generated)
├── feature_columns.pkl                       # Feature columns list (generated)
└── encoders.pkl                              # Label encoders mapping (generated)
```

## Setup & Execution

### 1. Install Dependencies
Install all required Python packages from the project root:
```bash
pip install -r requirements.txt
```

### 2. Preprocess the Dataset
Clean the raw data, perform conversions, and save it to the SQLite database:
```bash
python preprocess.py
```

### 3. Train and Compare Models
Train multiple classifiers on the preprocessed dataset and save the highest performing model (based on F1-Score):
```bash
python train.py
```

### 4. Run the API Server
Start the local Flask development server (running on port `5000`):
```bash
python app.py
```

## Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `0.8169` | `0.6803` | `0.5818` | **`0.6272`** |
| **Gradient Boosting** | `0.8055` | `0.6713` | `0.5201` | `0.5861` |
| **Random Forest** | `0.7956` | `0.6592` | `0.4718` | `0.5500` |

The **Logistic Regression** model was selected as the best model because it achieved the highest F1-Score of **0.6272** on the test set.


## API Usage

### Health Check Endpoint
* **Endpoint:** `GET /health`
* **Response:**
  ```json
  {
    "status": "ok"
  }
  ```

### Prediction Endpoint
* **Endpoint:** `POST /predict`
* **Headers:** `Content-Type: application/json`
* **Request JSON Body Example:**
  ```json
  {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }
  ```
* **Response JSON Example:**
  ```json
  {
    "churn_prediction": "Yes",
    "churn_probability": 0.61
  }
  ```

---
**Author:** Venkata Sesha Sainath Biruduraju
