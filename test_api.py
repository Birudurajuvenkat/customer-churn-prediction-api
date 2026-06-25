# Equivalent curl command:
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"gender\": \"Female\", \"SeniorCitizen\": 0, \"Partner\": \"Yes\", \"Dependents\": \"No\", \"tenure\": 1, \"PhoneService\": \"No\", \"MultipleLines\": \"No phone service\", \"InternetService\": \"DSL\", \"OnlineSecurity\": \"No\", \"OnlineBackup\": \"Yes\", \"DeviceProtection\": \"No\", \"TechSupport\": \"No\", \"StreamingTV\": \"No\", \"StreamingMovies\": \"No\", \"Contract\": \"Month-to-month\", \"PaperlessBilling\": \"Yes\", \"PaymentMethod\": \"Electronic check\", \"MonthlyCharges\": 29.85, \"TotalCharges\": 29.85}"

import requests
import json

def main():
    url = "http://127.0.0.1:5000/predict"
    
    # Realistic customer payload based on the Telco Customer Churn dataset
    payload = {
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
    
    print("Sending POST request to:", url)
    print("Payload:")
    print(json.dumps(payload, indent=2))
    print("-" * 40)
    
    try:
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print("An error occurred during request:", e)

if __name__ == "__main__":
    main()
