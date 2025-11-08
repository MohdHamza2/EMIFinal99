import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Paths to the saved artifacts
MODEL_PATH = Path("models/best_classification_model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")

# Load the preprocessor and trained model
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)

# ✅ Example input sample (update values as per your dataset)
sample = pd.DataFrame([{
    "age": "30-35",
    "gender": "Male",
    "marital_status": "Married",
    "education": "Graduate",
    "employment_type": "Salaried",
    "company_type": "Private",
    "house_type": "Rented",
    "existing_loans": "Yes",
    "bank_balance": "Medium",
    "emi_scenario": "Stable",
    "years_of_employment": 5,
    "monthly_rent": 15000,
    "family_size": 4,
    "dependents": 2,
    "school_fees": 10000,
    "college_fees": 0,
    "travel_expenses": 4000,
    "groceries_utilities": 12000,
    "other_monthly_expenses": 5000,
    "current_emi_amount": 10000,
    "credit_score": 720,
    "emergency_fund": 20000,
    "requested_amount": 500000,
    "requested_tenure": 24,
    "max_monthly_emi": 15000
}])

# Preprocess the input data
processed_sample = preprocessor.transform(sample)

# Get predictions
prediction = model.predict(processed_sample)

print("\n✅ Model Prediction:")
print(prediction[0])

# Check if the model supports probability outputs
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(processed_sample)
    classes = model.classes_
    print("\n📊 Prediction Probabilities:")
    for cls, prob in zip(classes, probabilities[0]):
        print(f"{cls}: {prob*100:.2f}%")
else:
    print("\n⚠️ This model does not support probability outputs.")
