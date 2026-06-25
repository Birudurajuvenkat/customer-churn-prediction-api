from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Paths to the saved artifacts relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")


# Load the model, features, and encoders on startup
try:
    print("Loading model and preprocessing artifacts...")
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print("Artifacts loaded successfully.")
    print("Expected feature columns:", feature_columns)
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model, feature_columns, encoders = None, None, None

@app.route('/health', methods=['GET'])
def health():
    """GET endpoint to check API health status."""
    return jsonify({"status": "ok"}), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Customer Churn Prediction API is running.",
        "status": "success"
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """POST endpoint to predict churn for a single customer."""
    if not model or not feature_columns or not encoders:
        return jsonify({"error": "Model or preprocessing artifacts are not loaded."}), 500
        
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be a valid JSON object."}), 400
        
    # Check for missing required fields
    missing_fields = [field for field in feature_columns if field not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields in the input data.",
            "missing_fields": missing_fields
        }), 400
        
    try:
        # Preprocess input data to match the expected format
        processed_data = {}
        for col in feature_columns:
            val = data[col]
            
            # If the column has a label encoder, use it to transform the categorical string value
            if col in encoders:
                val_str = str(val).strip()
                le = encoders[col]
                # Check if value is one of the categories learned during training
                if val_str in le.classes_:
                    processed_data[col] = le.transform([val_str])[0]
                else:
                    return jsonify({
                        "error": f"Invalid value '{val}' for field '{col}'. Expected one of: {list(le.classes_)}"
                    }), 400
            else:
                # Convert numeric fields to appropriate types
                if col in ['MonthlyCharges', 'TotalCharges']:
                    processed_data[col] = float(val)
                elif col in ['tenure', 'SeniorCitizen']:
                    processed_data[col] = int(val)
                else:
                    processed_data[col] = val
                    
        # Convert processed dictionary to a DataFrame to ensure exact column order
        input_df = pd.DataFrame([processed_data])[feature_columns]
        
        # Make prediction and get class probability of churn (class 1)
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        
        churn_pred_str = "Yes" if prediction == 1 else "No"
        
        return jsonify({
            "churn_prediction": churn_pred_str,
            "churn_probability": round(float(proba), 2)
        }), 200
        
    except ValueError as val_err:
        return jsonify({"error": f"Value conversion error: {str(val_err)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the application on port 5000 with debug=True
    app.run(host='0.0.0.0', port=5000, debug=True)
