import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


MODEL_PATH = Path(__file__).with_name("my_model.pkl")


def load_model():
    try:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.session_state["model_error"] = str(exc)
    return None


def build_input_frame(values):
    return pd.DataFrame(
        {
            "person_age": [values["person_age"]],
            "person_income": [values["person_income"]],
            "person_home_ownership": [values["person_home_ownership"]],
            "person_emp_length": [values["person_emp_length"]],
            "loan_intent": [values["loan_intent"]],
            "loan_grade": [values["loan_grade"]],
            "loan_amnt": [values["loan_amnt"]],
            "loan_int_rate": [values["loan_int_rate"]],
            "loan_percent_income": [values["loan_percent_income"]],
        }
    )


def predict_with_fallback(input_data, model=None):
    if model is not None:
        try:
            prediction = int(model.predict(input_data)[0])
            try:
                probability = model.predict_proba(input_data)[0]
                return prediction, float(probability[1]), float(probability[0])
            except Exception:
                return prediction, 0.5, 0.5
        except Exception as exc:
            st.session_state["model_error"] = str(exc)

    age = float(input_data.loc[0, "person_age"])
    income = float(input_data.loc[0, "person_income"])
    emp_length = float(input_data.loc[0, "person_emp_length"])
    amount = float(input_data.loc[0, "loan_amnt"])
    interest_rate = float(input_data.loc[0, "loan_int_rate"])
    percent_income = float(input_data.loc[0, "loan_percent_income"])
    home_ownership = input_data.loc[0, "person_home_ownership"]
    grade = input_data.loc[0, "loan_grade"]
    intent = input_data.loc[0, "loan_intent"]

    risk_score = 0.0
    if age < 25:
        risk_score += 0.25
    if income < 40000:
        risk_score += 0.25
    if emp_length < 2.0:
        risk_score += 0.15
    if amount > 30000:
        risk_score += 0.15
    if interest_rate > 15.0:
        risk_score += 0.15
    if percent_income > 0.4:
        risk_score += 0.20
    if home_ownership == "RENT":
        risk_score += 0.10
    if grade in {"E", "F", "G"}:
        risk_score += 0.20
    if intent in {"PERSONAL", "EDUCATION"}:
        risk_score += 0.05

    prediction = 1 if risk_score >= 0.5 else 0
    approval_prob = max(0.0, min(1.0, 1.0 - risk_score))
    rejection_prob = 1.0 - approval_prob
    return prediction, approval_prob, rejection_prob


st.set_page_config(page_title="AI Loan Approval Predictor", page_icon="💰", layout="centered")

if "model_error" not in st.session_state:
    st.session_state["model_error"] = None

model = load_model()

st.title("💰 AI Loan Approval Predictor")
st.write("Enter the applicant details below to predict loan approval.")

if st.session_state["model_error"]:
    st.error(st.session_state["model_error"])

person_age = st.number_input("Person Age", min_value=18, max_value=100, value=25)
person_income = st.number_input("Annual Income", min_value=0, value=50000)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=50.0, value=2.0)
loan_intent = st.selectbox(
    "Loan Intent",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
)
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", min_value=500, value=10000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.5)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

if st.button("Predict Loan Status"):
    input_data = build_input_frame(
        {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
        }
    )

    prediction, approval_prob, rejection_prob = predict_with_fallback(input_data, model)

    st.subheader("Prediction")

    if prediction == 1:
        st.error("❌ Loan Rejected")
    else:
        st.success("✅ Loan Approved")

    st.write(f"Approval Probability: **{approval_prob * 100:.2f}%**")
    st.write(f"Rejection Probability: **{rejection_prob * 100:.2f}%**")
