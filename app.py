import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path


# Set Page Configuration
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide"
)



MODEL_PATH = Path(__file__).with_name("my_model.pkl")


def load_model():
    try:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.session_state["model_error"] = str(exc)
    return None
# ---------------------------------------------------------
# 2. User Interface Inputs
# ---------------------------------------------------------
st.title("💳 Credit Risk & Default Prediction")
st.write("Enter applicant details below to assess loan default risk.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Applicant Profile")
    person_age = st.number_input("Age", min_value=18, max_value=100, value=28, step=1)
    person_income = st.number_input("Annual Income ($)", min_value=0.0, value=55000.0, step=1000.0)
    person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    st.subheader("💰 Loan Information")
    loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500.0, value=10000.0, step=500.0)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=11.0, step=0.1)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"])

with col3:
    st.subheader("📜 Credit History")
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    cb_person_default_on_file = st.selectbox("Historical Default on File?", ["N", "Y"])

# ---------------------------------------------------------
# 3. Feature Engineering & Preprocessing
# ---------------------------------------------------------
# Calculated Feature
loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0

# Base DataFrame matching continuous & binary flags
raw_data = {
    'person_age': float(person_age),
    'person_income': float(person_income),
    'person_emp_length': float(person_emp_length),
    'loan_amnt': float(loan_amnt),
    'loan_int_rate': float(loan_int_rate),
    'loan_percent_income': float(loan_percent_income),
    'cb_person_default_on_file_Y': 1.0 if cb_person_default_on_file == "Y" else 0.0,
}

# Explicitly map all 21 model features in exact training order
model_features = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_default_on_file_Y',
    'person_home_ownership_OTHER',
    'person_home_ownership_OWN',
    'person_home_ownership_RENT',
    'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL',
    'loan_intent_PERSONAL',
    'loan_intent_VENTURE',
    'loan_grade_B',
    'loan_grade_C',
    'loan_grade_D',
    'loan_grade_E',
    'loan_grade_F',
    'loan_grade_G'
]

# One-Hot Encoding for Home Ownership
raw_data['person_home_ownership_OTHER'] = 1.0 if person_home_ownership == 'OTHER' else 0.0
raw_data['person_home_ownership_OWN'] = 1.0 if person_home_ownership == 'OWN' else 0.0
raw_data['person_home_ownership_RENT'] = 1.0 if person_home_ownership == 'RENT' else 0.0

# One-Hot Encoding for Loan Intent
raw_data['loan_intent_EDUCATION'] = 1.0 if loan_intent == 'EDUCATION' else 0.0
raw_data['loan_intent_HOMEIMPROVEMENT'] = 1.0 if loan_intent == 'HOMEIMPROVEMENT' else 0.0
raw_data['loan_intent_MEDICAL'] = 1.0 if loan_intent == 'MEDICAL' else 0.0
raw_data['loan_intent_PERSONAL'] = 1.0 if loan_intent == 'PERSONAL' else 0.0
raw_data['loan_intent_VENTURE'] = 1.0 if loan_intent == 'VENTURE' else 0.0

# One-Hot Encoding for Loan Grade (Note: Grade A is implied reference level when all are 0)
for grade in ['B', 'C', 'D', 'E', 'F', 'G']:
    raw_data[f'loan_grade_{grade}'] = 1.0 if loan_grade == grade else 0.0

# Build final DataFrame ensuring column sequence matches model metadata
input_df = pd.DataFrame([raw_data])[model_features]

# ---------------------------------------------------------
# 4. Prediction Execution
# ---------------------------------------------------------
st.markdown("---")

if st.button("Predict Loan Default Risk", type="primary", use_container_width=True):
    # Obtain prediction probability for positive class (Default = 1)
    probabilities = my_model.predict_proba(input_df)[0]
    default_prob = probabilities[1]
    
    st.subheader("Assessment Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(
            label="Calculated Default Risk",
            value=f"{default_prob * 100:.2f}%"
        )
        st.progress(float(default_prob))

    with res_col2:
        # Decision threshold set at standard 50% (adjust if tuned differently)
        if default_prob >= 0.50:
            st.error("⚠️ **HIGH RISK**: Application flagged for potential default.")
        elif default_prob >= 0.30:
            st.warning("⚡ **MODERATE RISK**: Requires manual underwriting / verification.")
        else:
            st.success("✅ **LOW RISK**: Application satisfies standard low-risk threshold.")
