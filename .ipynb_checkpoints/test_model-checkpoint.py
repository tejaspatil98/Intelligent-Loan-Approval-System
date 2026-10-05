import pandas as pd
import numpy as np
import joblib

# 1. Load exported model and scaler
model = joblib.load('naive_bayes_model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_user_input(data_dict):
    # Squared features
    dti_sq = data_dict['DTI_Ratio'] ** 2
    credit_score_sq = data_dict['Credit_Score'] ** 2
    
    # Construct complete feature row matching X_train
    row = {
        'Applicant_Income': data_dict['Applicant_Income'],
        'Coapplicant_Income': data_dict['Coapplicant_Income'],
        'Age': data_dict['Age'],
        'Dependents': data_dict['Dependents'],
        'Existing_Loans': data_dict['Existing_Loans'],
        'Savings': data_dict['Savings'],
        'Collateral_Value': data_dict['Collateral_Value'],
        'Loan_Amount': data_dict['Loan_Amount'],
        'Loan_Term': data_dict['Loan_Term'],
        'Education_Level': 1 if data_dict['Education_Level'] == 'Graduate' else 0,
        
        # Employment Status
        'Employment_Status_Salaried': 1 if data_dict['Employment_Status'] == 'Salaried' else 0,
        'Employment_Status_Self-employed': 1 if data_dict['Employment_Status'] == 'Self-employed' else 0,
        'Employment_Status_Unemployed': 1 if data_dict['Employment_Status'] == 'Unemployed' else 0,
        
        # Marital Status
        'Marital_Status_Single': 1 if data_dict['Marital_Status'] == 'Single' else 0,
        
        # Loan Purpose
        'Loan_Purpose_Car': 1 if data_dict['Loan_Purpose'] == 'Car' else 0,
        'Loan_Purpose_Education': 1 if data_dict['Loan_Purpose'] == 'Education' else 0,
        'Loan_Purpose_Home': 1 if data_dict['Loan_Purpose'] == 'Home' else 0,
        'Loan_Purpose_Personal': 1 if data_dict['Loan_Purpose'] == 'Personal' else 0,
        
        # Property Area
        'Property_Area_Semiurban': 1 if data_dict['Property_Area'] == 'Semiurban' else 0,
        'Property_Area_Urban': 1 if data_dict['Property_Area'] == 'Urban' else 0,
        
        # Gender
        'Gender_Male': 1 if data_dict['Gender'] == 'Male' else 0,
        
        # Employer Category
        'Employer_Category_Government': 1 if data_dict['Employer_Category'] == 'Government' else 0,
        'Employer_Category_MNC': 1 if data_dict['Employer_Category'] == 'MNC' else 0,
        'Employer_Category_Private': 1 if data_dict['Employer_Category'] == 'Private' else 0,
        'Employer_Category_Unemployed': 1 if data_dict['Employer_Category'] == 'Unemployed' else 0,
        
        # Engineered Features
        'DTI_Ratio_sq': dti_sq,
        'Credit_Score_sq': credit_score_sq
    }
    
    df = pd.DataFrame([row])
    
    # Reorder columns to strictly match scaler expectations
    if hasattr(scaler, 'feature_names_in_'):
        df = df[scaler.feature_names_in_]
        
    return df

# Test Applicant Profile
test_applicant = {
    'Applicant_Income': 29589.0,
    'Coapplicant_Income': 8041.0,
    'Age': 31,
    'Dependents': 0,
    'Existing_Loans': 0,
    'Savings': 11906.0,
    'Collateral_Value': 8150.0,
    'Loan_Amount': 29287.0,
    'Loan_Term': 12,
    'Education_Level': 'Graduate',
    'Employment_Status': 'Salaried',
    'Marital_Status': 'Single',
    'Loan_Purpose': 'Personal',
    'Property_Area': 'Urban',
    'Gender': 'Male',
    'Employer_Category': 'Private',
    'DTI_Ratio': 0.35,
    'Credit_Score': 730
}

# Run preprocessing & scale
df_input = preprocess_user_input(test_applicant)
scaled_input = scaler.transform(df_input)

# Predict Outcome
pred = model.predict(scaled_input)[0]
prob = model.predict_proba(scaled_input)[0][1]

status = "APPROVED" if (pred == 1 or str(pred).lower() in ['yes', '1', 'true']) else "REJECTED"
print(f"✅ Successful Run!")
print(f"Prediction: {status} | Confidence: {prob * 100:.2f}%")