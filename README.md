# 💰 AI-Based Loan Approval Predictor

A machine learning–powered web application that predicts whether a loan application is likely to be **approved** or **rejected**, based on applicant and loan attributes. The app is built with **Streamlit** for the front end and a trained **XGBoost classifier** for prediction, with a built-in rule-based fallback predictor to guarantee the app never fails to return a result.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Tech Stack](#-tech-stack)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Input Parameters](#-input-parameters)
- [How Prediction Works](#-how-prediction-works)
- [Known Issues & Limitations](#-known-issues--limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🧠 Overview

The **AI-Based Loan Approval Predictor** allows a user to enter an applicant's personal and financial details through a simple web interface and instantly receive a prediction on whether their loan would be **approved** or **rejected**, along with the associated approval and rejection probabilities.

The application is designed with reliability in mind: if the trained model file cannot be loaded for any reason, the app automatically switches to a transparent, rule-based scoring system so the user experience is never interrupted.

---

## ✨ Features

- 🖥️ Clean, interactive web UI built with Streamlit
- 🤖 Predictions powered by a trained **XGBoost** classification model
- 🛡️ Automatic **fallback predictor** if the model fails to load or fails during inference
- 📊 Displays both **approval probability** and **rejection probability**
- ⚡ Real-time, single-click prediction — no page reloads
- 🧩 Modular, readable Python code (separated data-building, prediction, and UI logic)

---

## 🏗️ Project Architecture

```
┌────────────────────┐
│   Streamlit UI      │   ← Collects applicant details via form inputs
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│ build_input_frame() │   ← Converts form inputs into a Pandas DataFrame
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│ predict_with_       │   ← Tries the trained ML model first
│ fallback()           │   ← Falls back to a rule-based scorer on failure
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│  Result Display      │   ← Shows Approved/Rejected + probabilities
└────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer            | Technology                     |
|-------------------|--------------------------------|
| Frontend/UI       | Streamlit                      |
| Data Handling     | Pandas                         |
| Machine Learning  | XGBoost (`XGBClassifier`)      |
| Model Persistence | Joblib                         |
| Language          | Python 3                       |

---

## 🤖 Model Details

The prediction model (`my_model.pkl`) is a **trained XGBoost Classifier** (`xgboost.sklearn.XGBClassifier`), serialized using `joblib`.

| Property | Value |
|---|---|
| Model Type | `XGBClassifier` (Gradient Boosted Trees) |
| Objective | `binary:logistic` |
| Number of Estimators | 100 |
| Learning Rate | 0.1 |
| Number of Input Features | 21 (one-hot encoded) |

**Expected model input features (post-encoding):**

```
person_age, person_income, person_emp_length, loan_amnt,
loan_int_rate, loan_percent_income,
person_home_ownership_OTHER, person_home_ownership_OWN, person_home_ownership_RENT,
loan_intent_EDUCATION, loan_intent_HOMEIMPROVEMENT, loan_intent_MEDICAL,
loan_intent_PERSONAL, loan_intent_VENTURE,
loan_grade_B, loan_grade_C, loan_grade_D, loan_grade_E, loan_grade_F, loan_grade_G,
cb_person_default_on_file_Y
```

> The model was trained on a **one-hot encoded** version of the dataset (21 columns), where each categorical field (home ownership, loan intent, loan grade, previous default status) is expanded into binary indicator columns.

---

## 📁 Project Structure

```
loan-approval-predictor/
│
├── app.py            # Main Streamlit application
├── my_model.pkl       # Trained XGBoost model (serialized with joblib)
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone or download the project

```bash
git clone <https://github.com/abdullah-2007-saeed/ai-loan-approval-predictor.git>
cd loan-approval-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas joblib xgboost scikit-learn
```

Or, using a `requirements.txt`:

```
streamlit
pandas
joblib
xgboost
scikit-learn
```

```bash
pip install -r requirements.txt
```

### 4. Ensure the model file is present

Place `my_model.pkl` in the **same directory** as `app.py` (the app locates it automatically using its own file path).

---

## ▶️ Usage

Run the Streamlit app from the project directory:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`) in your browser.

**Steps to get a prediction:**

1. Fill in the applicant's details (age, income, employment length, etc.)
2. Select the loan-specific details (intent, grade, amount, interest rate)
3. Click **"Predict Loan Status"**
4. View the prediction result along with approval/rejection probabilities

---

## 📝 Input Parameters

| Field | Type | Description |
|---|---|---|
| Person Age | Numeric | Applicant's age (18–100) |
| Annual Income | Numeric | Applicant's yearly income |
| Home Ownership | Categorical | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| Employment Length | Numeric | Years of employment (0–50) |
| Loan Intent | Categorical | `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| Loan Grade | Categorical | `A` through `G` |
| Loan Amount | Numeric | Requested loan amount |
| Interest Rate (%) | Numeric | Loan interest rate |
| Loan Percent Income | Numeric | Loan amount as a fraction of income (0–1) |

---

## 🔍 How Prediction Works

1. **Model Loading** — On startup, the app attempts to load `my_model.pkl` from disk using `joblib.load()`. If this fails, the error is captured and a warning banner is shown to the user.
2. **Input Assembly** — User inputs from the form are assembled into a single-row Pandas DataFrame.
3. **Primary Prediction (ML Model)** — The DataFrame is passed to the trained XGBoost model's `.predict()` and `.predict_proba()` methods to obtain a class label and confidence scores.
4. **Fallback Prediction (Rule-Based)** — If the model is unavailable or raises an exception during inference, a transparent, weighted rule-based risk score is calculated instead, using heuristics such as:
   - Age under 25 → +0.25 risk
   - Income under 40,000 → +0.25 risk
   - Employment length under 2 years → +0.15 risk
   - Loan amount over 30,000 → +0.15 risk
   - Interest rate over 15% → +0.15 risk
   - Loan-to-income ratio over 40% → +0.20 risk
   - Renting (vs. owning) → +0.10 risk
   - Loan grade E, F, or G → +0.20 risk
   - Personal/education loan intent → +0.05 risk

   A cumulative risk score of **0.5 or higher** results in a **Rejected** prediction; otherwise, the loan is predicted **Approved**.
   but it is removed bucause it decrease quality of model
5. **Result Display** — The predicted outcome and both probabilities are shown to the user.

---

## ⚠️ Known Issues & Limitations

- **Feature schema mismatch:** The trained model expects **21 one-hot encoded features** (e.g., `loan_grade_B`, `person_home_ownership_RENT`), while the current `build_input_frame()` function in `app.py` builds a **9-column DataFrame** with raw categorical values (e.g., `loan_grade = "B"`). This mismatch will cause `model.predict()` to raise an exception on every request, meaning **the app currently always uses the rule-based fallback predictor**, not the trained XGBoost model. To use the trained model correctly, the input DataFrame must be preprocessed with the same one-hot encoding used during training (including the `cb_person_default_on_file` field, which is not currently collected in the UI at all).
- The UI does not currently collect a **"previous default on file"** value, which the model was trained on (`cb_person_default_on_file_Y`).
- No input validation beyond basic numeric ranges (e.g., no cross-field business rule checks).
- No persistence/logging of past predictions.

---

## 🚀 Future Improvements

- Fix the preprocessing pipeline so form inputs are one-hot encoded to exactly match the model's 21 training features (ideally by saving and reusing the training-time `ColumnTransformer`/`OneHotEncoder` alongside the model).
- Add a "Previous Default on File" input field to the UI.
- Add model performance metrics (accuracy, precision, recall, ROC-AUC) to the app or documentation.
- Support multi-model comparison (e.g., Logistic Regression, Random Forest, XGBoost) with a selectable model dropdown.
- Add SHAP-based explainability to show why a loan was approved/rejected.
- Containerize the app with Docker for easier deployment.
- Add unit tests for `build_input_frame()` and `predict_with_fallback()`.

---

## 👤 Author

**Abdullah Saeed**
Tecveq

---

*This project was developed as part of an portfolio initiative to demonstrate an end-to-end machine learning web application, from model training to interactive deployment using Streamlit.*
