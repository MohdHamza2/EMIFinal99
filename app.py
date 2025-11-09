import sys, os
import gdown
from pathlib import Path

# --- Ensure gdown is available ---
try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_models():
    files = {
        "preprocessor.joblib": "https://drive.google.com/uc?id=1I-MrTta_UJBEdGwedMZ6CPhLo6V0CiTv",
        "label_encoder.joblib": "https://drive.google.com/uc?id=1gC7m50nG_l_-Bo4JprYC7xsWBYcWd6S7",
        "best_classification_model.joblib": "https://drive.google.com/uc?id=13T0g8YMf-q7EE3yjAxvUuR9rTkG7iBId",
        "best_regression_model.joblib": "https://drive.google.com/uc?id=1capauhMI6Hl8W7w-dyqDQRKgAF6k0J92",
    }
    for name, url in files.items():
        dest = MODELS_DIR / name
        if not dest.exists():
            print(f"📥 Downloading {name}...")
            gdown.download(url, str(dest), quiet=False)
        else:
            print(f"✅ {name} already exists.")
# Call this before load_models()
download_models()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px


# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import Paths
from predict import make_prediction, load_models


# ---------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------
st.set_page_config(
    page_title="AI EMI Prediction Assistant",
    layout="wide",
    page_icon="💰"
)

st.markdown(
    """
    <style>
        .big-font { font-size:22px !important; font-weight:600; }
        .metric-card {
            background: #111827;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------
# Load Models
# ---------------------------------------------
try:
    preprocessor, label_encoder, best_classification_model, best_regression_model = load_models()
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ---------------------------------------------
# Model Status Section (inserted here)
# ---------------------------------------------
st.sidebar.markdown("### 🧩 Model Status")

models_dir = Path(Paths.MODELS_DIR)

try:
    preprocessor_path = models_dir / "preprocessor.joblib"
    clf_path = models_dir / "best_classification_model.joblib"
    reg_path = models_dir / "best_regression_model.joblib"
    label_encoder_path = models_dir / "label_encoder.joblib"

    # Fetch metadata
    file_info = {
        "Preprocessor": preprocessor_path,
        "Classifier": clf_path,
        "Regressor": reg_path,
        "Label Encoder": label_encoder_path
    }

    for name, path in file_info.items():
        if path.exists():
            modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))
            st.sidebar.write(f"✅ **{name}** loaded  \n📅 *Last updated:* {modified_time}")
        else:
            st.sidebar.error(f"❌ {name} not found")

    # Extra: Feature info
    if hasattr(preprocessor, "feature_names_in_"):
        st.sidebar.write(f"📊 **Features Used:** {len(preprocessor.feature_names_in_)}")

except Exception as e:
    st.sidebar.error(f"⚠️ Error fetching model info: {e}")

st.sidebar.header("⚙️ Settings")
st.sidebar.success("Models loaded successfully ✅")

# ---------------------------------------------
# Input Form
# ---------------------------------------------
st.title("💼 EMI Eligibility & Prediction Dashboard")

with st.form("emi_form"):
    st.subheader("📋 Enter Applicant Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 70, 32)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        education = st.selectbox("Education", ["Graduate", "Postgraduate", "Undergraduate"])
    with c2:
        monthly_salary = st.number_input("Monthly Salary (₹)", 5000, 200000, 20000, step=500)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
        company_type = st.selectbox("Company Type", ["Private", "Government", "Startup"])
        years_of_employment = st.slider("Years of Employment", 0, 40, 5)
    with c3:
        house_type = st.selectbox("House Type", ["Rented", "Owned"])
        monthly_rent = st.number_input("Monthly Rent (₹)", 0, 50000, 2500, step=500)
        family_size = st.number_input("Family Size", 1, 10, 4)
        dependents = st.number_input("Dependents", 0, 6, 2)

    c4, c5, c6 = st.columns(3)
    with c4:
        school_fees = st.number_input("School Fees (₹)", 0, 50000, 1000)
        college_fees = st.number_input("College Fees (₹)", 0, 100000, 500)
        travel_expenses = st.number_input("Travel Expenses (₹)", 0, 20000, 700)
    with c5:
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 30000, 1500)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 30000, 800)
        existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
    with c6:
        current_emi_amount = st.number_input("Current EMI (₹)", 0, 50000, 0)
        bank_balance = st.number_input("Bank Balance (₹)", 0, 1000000, 12000)
        credit_score = st.number_input("Credit Score", 300, 900, 710)
        emergency_fund = st.number_input("Emergency Fund (₹)", 0, 500000, 5000)

    st.markdown("---")
    st.subheader("💳 Loan Request Details")

    c7, c8 = st.columns(2)
    with c7:
        requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 1000000, 60000)
    with c8:
        requested_tenure = st.slider("Requested Tenure (Months)", 6, 60, 24)

    emi_scenario = st.selectbox("Scenario", ["Scenario_1", "Scenario_2", "Scenario_3"])

    submit = st.form_submit_button("🔮 Predict EMI Eligibility")

# ---------------------------------------------
# On Submit
# ---------------------------------------------
if submit:
    input_data = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "company_type": company_type,
        "years_of_employment": years_of_employment,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "bank_balance": bank_balance,
        "credit_score": credit_score,
        "emergency_fund": emergency_fund,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "emi_scenario": emi_scenario,
    }

    with st.spinner("Analyzing your financial profile... 💭"):
        time.sleep(1.2)
        result = make_prediction(input_data)

    st.success("✅ Prediction Completed!")

    # ---------------------------------------------
    # Results Display
    # ---------------------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Prediction Summary")
        st.metric("EMI Eligibility", result["emi_eligibility"])
        st.metric("Predicted EMI Amount", f"₹{result['predicted_emi_amount']}" if result["predicted_emi_amount"] != "N/A" else "N/A")

    with col2:
        # Credit Score Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Credit Strength"},
            gauge={'axis': {'range': [300, 900]},
                   'bar': {'color': "#06b6d4"},
                   'steps': [
                       {'range': [300, 600], 'color': "#dc2626"},
                       {'range': [600, 750], 'color': "#facc15"},
                       {'range': [750, 900], 'color': "#16a34a"}]}
        ))
        st.plotly_chart(fig_gauge, width="stretch")

    # ---------------------------------------------
    # AI Insights
    # ---------------------------------------------
    st.markdown("### 🧠 AI Insights")
    insights = []
    if monthly_rent / monthly_salary > 0.3:
        insights.append("🏠 Your rent exceeds 30% of your income — may affect eligibility.")
    if credit_score < 650:
        insights.append("⚠️ Low credit score — improve it for better loan chances.")
    if emergency_fund < monthly_salary * 0.5:
        insights.append("💰 Emergency fund is low — consider saving at least 50% of your salary.")
    if not insights:
        insights.append("✅ Your financial profile looks healthy for EMI approval.")
    for tip in insights:
        st.info(tip)

    # ---------------------------------------------
    # EMI Simulation (if eligible)
    # ---------------------------------------------
    #if result["emi_eligibility"].lower() == "eligible":
        #st.markdown("### 🎯 EMI Simulation")
        #sim_amount = st.slider("Simulate Loan Amount (₹)", 10000, 300000, requested_amount, step=5000)
        #sim_tenure = st.slider("Simulate Tenure (Months)", 6, 60, requested_tenure)

        #sim_emi = (float(result["predicted_emi_amount"]) * (sim_amount / requested_amount)) * (requested_tenure / sim_tenure)
        #st.metric("Simulated EMI", f"₹{sim_emi:,.2f}")

        # Plot EMI trend
        #tenures = np.arange(6, 61, 6)
        #emi_values = [float(result["predicted_emi_amount"]) * (requested_tenure / t) for t in tenures]
        #fig_line = px.line(x=tenures, y=emi_values, title="📈 EMI vs Tenure (Months)",
                           #labels={'x': 'Tenure (Months)', 'y': 'EMI Amount (₹)'})
        #st.plotly_chart(fig_line, width="stretch")

    # ---------------------------------------------
    # Smart Financial Tips
    # ---------------------------------------------
    st.markdown("### 💡 Smart Financial Tips")
    tips = []
    if credit_score < 650:
        tips.append("Increase your credit score by paying bills and EMIs on time.")
    if monthly_rent / monthly_salary > 0.3:
        tips.append("Try to reduce rent-to-income ratio below 30%.")
    if emergency_fund < monthly_salary * 0.5:
        tips.append("Increase your emergency fund for better financial resilience.")
    if existing_loans == "Yes":
        tips.append("Consider consolidating multiple loans to reduce EMI burden.")

    if tips:
        for t in tips:
            st.write(f"✔️ {t}")
    else:
        st.success("🌟 Your financial health appears strong!")

