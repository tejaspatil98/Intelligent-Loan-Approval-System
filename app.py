import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="SecureTrust Bank - Loan Prediction", page_icon="🏦", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('naive_bayes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error("Model artifacts missing! Ensure 'naive_bayes_model.pkl' and 'scaler.pkl' are in the project folder.")
    st.stop()

st.title("🏦 SecureTrust Bank - Loan Decision System")
st.caption("Automated Loan Approval Evaluation Powered by Machine Learning")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Applicant Details")
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0.0, value=25000.0, step=1000.0)
    coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0.0, value=5000.0, step=1000.0)
    age = st.slider("Age", 18, 70, 32)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    dependents = st.number_input("Dependents", 0, 10, 0)

with col2:
    st.subheader("📊 Credit & Risk Credentials")
    credit_score = st.slider("Credit Score", 300, 900, 720)
    dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", 0.0, 1.0, 0.25, 0.01)
    existing_loans = st.number_input("Existing Active Loans", 0, 10, 0)
    savings = st.number_input("Total Savings (₹)", min_value=0.0, value=15000.0, step=1000.0)
    collateral_value = st.number_input("Collateral Value (₹)", min_value=0.0, value=20000.0, step=1000.0)

with col3:
    st.subheader("📋 Loan & Employment Info")
    loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=1000.0, value=25000.0, step=1000.0)
    loan_term = st.selectbox("Tenure (Months)", [12, 24, 36, 48, 60, 72, 84])
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Car", "Business", "Education"])
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    employer_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")

if st.button("Evaluate Application Risk", type="primary"):
    dti_sq = dti_ratio ** 2
    credit_score_sq = credit_score ** 2

    # Feature Dictionary
    input_dict = {
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Age': float(age),
        'Dependents': float(dependents),
        'Existing_Loans': float(existing_loans),
        'Savings': savings,
        'Collateral_Value': collateral_value,
        'Loan_Amount': loan_amount,
        'Loan_Term': float(loan_term),
        'Education_Level': 1.0 if education_level == "Graduate" else 0.0,
        
        'Employment_Status_Salaried': 1.0 if employment_status == "Salaried" else 0.0,
        'Employment_Status_Self-employed': 1.0 if employment_status == "Self-employed" else 0.0,
        'Employment_Status_Unemployed': 1.0 if employment_status == "Unemployed" else 0.0,
        
        'Marital_Status_Single': 1.0 if marital_status == "Single" else 0.0,
        
        'Loan_Purpose_Car': 1.0 if loan_purpose == "Car" else 0.0,
        'Loan_Purpose_Education': 1.0 if loan_purpose == "Education" else 0.0,
        'Loan_Purpose_Home': 1.0 if loan_purpose == "Home" else 0.0,
        'Loan_Purpose_Personal': 1.0 if loan_purpose == "Personal" else 0.0,
        
        'Property_Area_Semiurban': 1.0 if property_area == "Semiurban" else 0.0,
        'Property_Area_Urban': 1.0 if property_area == "Urban" else 0.0,
        
        'Gender_Male': 1.0 if gender == "Male" else 0.0,
        
        'Employer_Category_Government': 1.0 if employer_category == "Government" else 0.0,
        'Employer_Category_MNC': 1.0 if employer_category == "MNC" else 0.0,
        'Employer_Category_Private': 1.0 if employer_category == "Private" else 0.0,
        'Employer_Category_Unemployed': 1.0 if employer_category == "Unemployed" else 0.0,
        
        'DTI_Ratio_sq': dti_sq,
        'Credit_Score_sq': credit_score_sq
    }

    input_df = pd.DataFrame([input_dict])

    # Reorder columns to strictly match feature order recorded at fit time
    if hasattr(scaler, 'feature_names_in_'):
        input_df = input_df[scaler.feature_names_in_]

    try:
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1 or str(prediction).lower() in ['yes', '1', 'true']:
                st.success("✅ **STATUS: LOAN APPROVED**")
                st.balloons()
            else:
                st.error("❌ **STATUS: LOAN REJECTED**")

        with res_col2:
            st.metric(label="Approval Probability", value=f"{probability * 100:.2f}%")
            st.progress(float(probability))

    except Exception as err:
        st.error(f"Inference error: {err}")