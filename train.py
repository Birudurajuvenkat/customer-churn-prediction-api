import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "churn.db")

    
    # 1. Load the cleaned data from churn.db (table "customers")
    print("--- 1. Loading data from SQLite database ---")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run preprocess.py first.")
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    print(f"Loaded {len(df)} records from database.\n")
    
    # 2. Split features (X) and target (y = Churn)
    print("--- 2. Splitting data into train and test sets ---")
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found in database table.")
        
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # 80/20 train/test split, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}\n")
    
    # 3. Train three classification models
    print("--- 3. Training models ---")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    # Dictionary to hold results
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # 4. Calculate metrics on the test set
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "model_obj": model,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Confusion Matrix": cm
        }
        
        print(f"{name} trained. F1-Score: {f1:.4f}")
    print("\n")
    
    # 4. Print confusion matrices
    print("--- 4. Confusion Matrices on Test Set ---")
    for name, metrics in results.items():
        print(f"\n{name} Confusion Matrix:")
        print(metrics["Confusion Matrix"])
    print("\n")
    
    # 5. Print clean comparison table showing all metrics
    print("--- 5. Model Comparison Table ---")
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            "Model": name,
            "Accuracy": f"{metrics['Accuracy']:.4f}",
            "Precision": f"{metrics['Precision']:.4f}",
            "Recall": f"{metrics['Recall']:.4f}",
            "F1-Score": f"{metrics['F1-Score']:.4f}"
        })
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print("\n")
    
    # 6. Automatically select model with highest F1-score
    print("--- 6. Selecting the Best Model ---")
    best_model_name = max(results, key=lambda name: results[name]["F1-Score"])
    best_metrics = results[best_model_name]
    print(f"Best model selected: {best_model_name}")
    print(f"Metrics - F1: {best_metrics['F1-Score']:.4f}, Accuracy: {best_metrics['Accuracy']:.4f}\n")
    
    # 7. Save the best model, feature column names, and encoders
    print("--- 7. Saving model artifacts ---")
    # Save best model
    model_save_path = os.path.join(base_dir, "model.pkl")
    joblib.dump(best_metrics["model_obj"], model_save_path)

    print(f"Best model saved to: {model_save_path}")
    
    # Save feature columns
    features_save_path = os.path.join(base_dir, "feature_columns.pkl")
    feature_cols = list(X.columns)
    joblib.dump(feature_cols, features_save_path)
    print(f"Feature column names saved to: {features_save_path}")
    
    # Save label encoders (verifying from preprocessing step)
    src_encoders_path = os.path.join(base_dir, "encoders.pkl")
    dest_encoders_path = os.path.join(base_dir, "encoders.pkl")
    if os.path.exists(src_encoders_path):
        encoders = joblib.load(src_encoders_path)
        joblib.dump(encoders, dest_encoders_path)
        print(f"LabelEncoders saved to: {dest_encoders_path}")
    else:
        print("Warning: encoders.pkl was not found at expected path. Make sure preprocess.py ran successfully.")
    print("\n")
    
    # 8. Print which model was selected as best and why (based on F1-score)
    print("--- 8. Selection Summary ---")
    print(f"Selected Model: {best_model_name}")
    print(f"Reason: Out of the trained models, '{best_model_name}' achieved the highest F1-Score of {best_metrics['F1-Score']:.4f} on the test set.")
    print("This indicates the best balance between Precision and Recall for predicting churn customers.")

if __name__ == "__main__":
    main()
