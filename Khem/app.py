import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

# ===========================
# Load trained model & encoder
# ===========================
# Make sure you have already saved your trained XGBoost model as model.json or model.pkl
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)  # If saved as Booster object, use xgb.Booster.load_model()

# ===========================
# Streamlit UI
# ===========================
st.title("Insurance Premium Prediction App")

st.markdown("### Enter the details below to predict the Premium Amount")

# ---------------------------
# Collect user inputs
# ---------------------------
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
num_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])
occupation = st.text_input("Occupation")
health_score = st.number_input("Health Score", min_value=0.0, step=0.1)
location = st.text_input("Location")
policy_type = st.text_input("Policy Type")
previous_claims = st.number_input("Previous Claims", min_value=0.0, step=0.1)
vehicle_age = st.number_input("Vehicle Age", min_value=0.0, step=0.1)
credit_score = st.number_input("Credit Score", min_value=0.0, step=1.0)
insurance_duration = st.number_input("Insurance Duration", min_value=0.0, step=0.1)
policy_start_date = st.date_input("Policy Start Date")
customer_feedback = st.text_input("Customer Feedback")
smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
exercise_frequency = st.selectbox("Exercise Frequency", ["Never", "Rarely", "Occasionally", "Regularly"])
property_type = st.text_input("Property Type")
age_group = st.selectbox("Age Group", ["Young", "Middle-aged", "Senior"])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Premium Amount"):
    # Create DataFrame for model input
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Annual Income": annual_income,
        "Marital Status": marital_status,
        "Number of Dependents": num_dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Policy Start Date": str(policy_start_date),
        "Customer Feedback": customer_feedback,
        "Smoking Status": smoking_status,
        "Exercise Frequency": exercise_frequency,
        "Property Type": property_type,
        "Age Group": age_group
    }])

    # Preprocess categorical columns (same as training)
    cat_columns = input_df.select_dtypes(include="object").columns
    for col in cat_columns:
        input_df[col] = input_df[col].astype("category").cat.codes

    # Convert to DMatrix for XGBoost
    dmatrix_input = xgb.DMatrix(input_df, enable_categorical=True)

    # Make prediction
    prediction = model.predict(dmatrix_input)
    st.success(f"Predicted Premium Amount: {prediction[0]:.2f}")
