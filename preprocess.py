import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sqlite3
import joblib


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    db_path = os.path.join(base_dir, "churn.db")

    
    # 1. Load the CSV using pandas
    print("--- 1. Loading the CSV file ---")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}\n")
    
    # 2. Print df.info() and df.isnull().sum()
    print("--- 2. Data Quality Check ---")
    print("DataFrame Info:")
    df.info()
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    print("\n")
    
    # 3. Drop the 'customerID' column
    print("--- 3. Dropping 'customerID' column ---")
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        print("Dropped 'customerID'.\n")
    else:
        print("'customerID' column not found.\n")
        
    # 4. Convert the 'TotalCharges' column to numeric, coercing errors, and fill any resulting missing values with the median.
    print("--- 4. Preprocessing 'TotalCharges' column ---")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_count = df['TotalCharges'].isnull().sum()
    print(f"Number of missing/coerced values in 'TotalCharges': {missing_count}")
    
    median_val = df['TotalCharges'].median()
    print(f"Filling missing values with median: {median_val}")
    df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
    print(f"Number of missing values in 'TotalCharges' after imputation: {df['TotalCharges'].isnull().sum()}\n")
    
    # 5. Encode the target column 'Churn' as 1 for "Yes" and 0 for "No".
    print("--- 5. Encoding 'Churn' target column ---")
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        print("Churn target column encoded successfully.\n")
    else:
        print("'Churn' column not found.\n")
        
    # 6. Encode all remaining categorical columns using LabelEncoder from sklearn, and keep track of which columns were encoded.
    print("--- 6. Encoding remaining categorical columns ---")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {categorical_cols}")
    
    encoded_columns = []
    label_encoders = {}
    for col in categorical_cols:
        print(f"Encoding column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        encoded_columns.append(col)
        
    print(f"Successfully encoded {len(encoded_columns)} columns: {encoded_columns}\n")
    
    # Save encoders to encoders.pkl
    encoders_path = os.path.join(base_dir, "encoders.pkl")

    joblib.dump(label_encoders, encoders_path)
    print(f"Label encoders saved to '{encoders_path}'.\n")

    
    # 7. Save the cleaned dataframe into a SQLite database file called churn.db (in the project root), in a table called "customers".
    print("--- 7. Saving dataframe to SQLite database 'churn.db' ---")
    conn = sqlite3.connect(db_path)
    df.to_sql('customers', conn, if_exists='replace', index=False)
    conn.commit()
    print("Cleaned dataframe saved to table 'customers' in 'churn.db'.\n")
    
    # 8. After saving, run and print the results of these 2 SQL queries against churn.db:
    # a. Average MonthlyCharges grouped by Churn status
    # b. Count of customers grouped by Contract type and Churn status
    print("--- 8. Running SQL queries against churn.db ---")
    
    print("Query A: Average MonthlyCharges grouped by Churn status")
    query_a = """
    SELECT Churn, AVG(MonthlyCharges) AS AvgMonthlyCharges
    FROM customers
    GROUP BY Churn;
    """
    df_query_a = pd.read_sql_query(query_a, conn)
    print(df_query_a.to_string(index=False))
    print("\n")
    
    print("Query B: Count of customers grouped by Contract type and Churn status")
    if 'Contract' in label_encoders:
        print("Contract encoding mapping:")
        for idx, label in enumerate(label_encoders['Contract'].classes_):
            print(f"  {idx} -> {label}")
    
    query_b = """
    SELECT Contract, Churn, COUNT(*) AS CustomerCount
    FROM customers
    GROUP BY Contract, Churn;
    """
    df_query_b = pd.read_sql_query(query_b, conn)
    print(df_query_b.to_string(index=False))
    print("\n")
    
    conn.close()
    print("Database connection closed. Preprocessing finished.")

if __name__ == "__main__":
    main()
