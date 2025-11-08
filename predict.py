import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Paths
import joblib
import pandas as pd
from pathlib import Path
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models():
    """Load all required models and preprocessing artifacts."""
    try:
        preprocessor = joblib.load(Path(Paths.MODELS_DIR) / "preprocessor.joblib")
        label_encoder = joblib.load(Path(Paths.MODELS_DIR) / "label_encoder.joblib")
        clf_model = joblib.load(Path(Paths.MODELS_DIR) / "best_classification_model.joblib")
        reg_model = joblib.load(Path(Paths.MODELS_DIR) / "best_regression_model.joblib")

        logger.info("✅ All models and preprocessors loaded successfully.")
        return preprocessor, label_encoder, clf_model, reg_model
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise


def make_prediction(input_data: dict):
    """Make EMI eligibility and EMI amount predictions for a given input."""
    preprocessor, label_encoder, clf_model, reg_model = load_models()

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    logger.info("📦 Input received for prediction.")

    # Ensure target column is not included
    if 'emi_eligibility' in input_df.columns:
        input_df = input_df.drop(columns=['emi_eligibility'])

    # Align input columns to match preprocessor
    expected_columns = preprocessor.feature_names_in_ if hasattr(preprocessor, "feature_names_in_") else None

    if expected_columns is not None:
        missing_cols = set(expected_columns) - set(input_df.columns)
        extra_cols = set(input_df.columns) - set(expected_columns)

        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df.drop(columns=list(extra_cols), errors="ignore")
        input_df = input_df[expected_columns]

    # Transform input
    X_processed = preprocessor.transform(input_df)

    import numpy as np

    # --- Handle classifier ---
    clf_n_features = getattr(clf_model, "n_features_in_", X_processed.shape[1])
    if X_processed.shape[1] > clf_n_features:
        logger.warning(f"⚠️ Trimming features for classifier from {X_processed.shape[1]} → {clf_n_features}")
        X_clf = X_processed[:, :clf_n_features]
    elif X_processed.shape[1] < clf_n_features:
        pad_width = clf_n_features - X_processed.shape[1]
        X_clf = np.pad(X_processed, ((0, 0), (0, pad_width)), mode='constant')
    else:
        X_clf = X_processed

    # Predict eligibility
    eligibility_pred = clf_model.predict(X_clf)
    eligibility_label = label_encoder.inverse_transform(eligibility_pred)[0]

    # --- Handle regression only if eligible ---
    emi_amount_pred = None
    if eligibility_label.lower() == "eligible":
        reg_n_features = getattr(reg_model, "n_features_in_", X_processed.shape[1])
        if X_processed.shape[1] > reg_n_features:
            logger.warning(f"⚠️ Trimming features for regressor from {X_processed.shape[1]} → {reg_n_features}")
            X_reg = X_processed[:, :reg_n_features]
        elif X_processed.shape[1] < reg_n_features:
            pad_width = reg_n_features - X_processed.shape[1]
            X_reg = np.pad(X_processed, ((0, 0), (0, pad_width)), mode='constant')
        else:
            X_reg = X_processed

        emi_amount_pred = round(reg_model.predict(X_reg)[0], 2)
        logger.info(f"💰 Predicted EMI Amount: {emi_amount_pred}")
    else:
        logger.info("❌ Not eligible — skipping EMI amount prediction.")

    logger.info("✅ Prediction completed successfully.")
    return {
        "emi_eligibility": eligibility_label,
        "predicted_emi_amount": emi_amount_pred if emi_amount_pred is not None else "N/A"
    }


if __name__ == "__main__":
    # Example input
    sample_input = {
        "age": 32,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Graduate",
        "monthly_salary": 20000,
        "employment_type": "Salaried",
        "company_type": "Private",
        "years_of_employment": 5,
        "house_type": "Rented",
        "monthly_rent": 2500,
        "family_size": 4,
        "dependents": 2,
        "school_fees": 1000,
        "college_fees": 500,
        "travel_expenses": 700,
        "groceries_utilities": 1500,
        "other_monthly_expenses": 800,
        "existing_loans": "No",
        "current_emi_amount": 100,
        "bank_balance": 120000,
        "credit_score": 610,
        "emergency_fund": 5000,
        "requested_amount": 60000,
        "requested_tenure": 24,
        "emi_scenario": "Scenario_1"
    }

    result = make_prediction(sample_input)
    print("\n🔮 Prediction Result:")
    if result["emi_eligibility"].lower() == "eligible":
        print(f"✅ EMI Eligibility: {result['emi_eligibility']}")
        print(f"💰 Predicted EMI Amount: {result['predicted_emi_amount']}")
    else:
        print(f"❌ EMI Eligibility: {result['emi_eligibility']}")
        print("💬 This customer is not eligible for an EMI plan.")

