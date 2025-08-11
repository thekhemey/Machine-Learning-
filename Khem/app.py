# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from datetime import datetime

# ===========================
# Load trained model & encoder
# ===========================
@st.cache_resource
def load_model():
    """Load the trained XGBoost model with error handling"""
    try:
        if os.path.exists("xgb_model.pkl"):
            with open("xgb_model.pkl", "rb") as file:
                model = pickle.load(file)
            return model, "pickle"
        elif os.path.exists("xgb_model.json"):
            model = xgb.Booster()
            model.load_model("xgb_model.json")
            return model, "json"
        else:
            st.error("Model file not found! Please ensure 'xgb_model.pkl' or 'xgb_model.json' exists.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# ===========================
# Feature preprocessing functions
# ===========================
def preprocess_input(input_df, training_columns=None):
    """
    Preprocess input data to match training format
    """
    try:
        # Handle date column
        if 'Policy Start Date' in input_df.columns:
            # Convert date to numerical features
            policy_date = pd.to_datetime(input_df['Policy Start Date'].iloc[0])
            input_df['Policy_Year'] = policy_date.year
            input_df['Policy_Month'] = policy_date.month
            input_df['Policy_Day'] = policy_date.day
            input_df = input_df.drop('Policy Start Date', axis=1)
        
        # Handle categorical columns
        categorical_mappings = {
            'Gender': {'Male': 0, 'Female': 1, 'Other': 2},
            'Marital Status': {'Single': 0, 'Married': 1, 'Divorced': 2, 'Widowed': 3},
            'Education Level': {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3, 'Other': 4},
            'Smoking Status': {'No': 0, 'Yes': 1},
            'Exercise Frequency': {'Never': 0, 'Rarely': 1, 'Occasionally': 2, 'Regularly': 3},
            'Age Group': {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
        }
        
        # Apply categorical mappings
        for col, mapping in categorical_mappings.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(mapping).fillna(0)
        
        # Handle text columns (simple encoding)
        text_columns = ['Occupation', 'Location', 'Policy Type', 'Customer Feedback', 'Property Type']
        for col in text_columns:
            if col in input_df.columns:
                # Simple hash-based encoding for text
                input_df[col] = input_df[col].astype(str).apply(lambda x: hash(x.lower()) % 1000)
        
        # Ensure all columns are numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        return input_df
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

# ===========================
# Streamlit UI
# ===========================
def main():
    st.set_page_config(page_title="Insurance Premium Predictor", page_icon="🛡️")
    
    st.title("🛡️ Insurance Premium Prediction App")
    st.markdown("### Enter the details below to predict the Premium Amount")
    
    # Load model
    model, model_type = load_model()
    if model is None:
        st.stop()
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
            education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])
            occupation = st.text_input("Occupation", value="Engineer")
            health_score = st.number_input("Health Score", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
            location = st.text_input("Location", value="Urban")
            policy_type = st.text_input("Policy Type", value="Comprehensive")
        
        with col2:
            previous_claims = st.number_input("Previous Claims ($)", min_value=0.0, value=0.0, step=100.0)
            vehicle_age = st.number_input("Vehicle Age (years)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
            credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=700.0, step=1.0)
            insurance_duration = st.number_input("Insurance Duration (years)", min_value=0.0, value=1.0, step=0.1)
            policy_start_date = st.date_input("Policy Start Date", value=datetime.now().date())
            customer_feedback = st.text_input("Customer Feedback", value="Satisfied")
            smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])
            exercise_frequency = st.selectbox("Exercise Frequency", ["Never", "Rarely", "Occasionally", "Regularly"])
            property_type = st.text_input("Property Type", value="House")
            age_group = st.selectbox("Age Group", ["Young", "Middle-aged", "Senior"])
        
        # Prediction button
        submitted = st.form_submit_button("🔮 Predict Premium Amount", use_container_width=True)
        
        if submitted:
            # Create input DataFrame
            input_data = {
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
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input
            processed_df = preprocess_input(input_df)
            
            if processed_df is not None:
                try:
                    # Make prediction based on model type
                    if model_type == "pickle":
                        # For scikit-learn style model
                        prediction = model.predict(processed_df)
                    else:
                        # For XGBoost Booster
                        dmatrix_input = xgb.DMatrix(processed_df)
                        prediction = model.predict(dmatrix_input)
                    
                    # Display result
                    premium_amount = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
                    
                    st.success(f"🎯 **Predicted Premium Amount: ${premium_amount:.2f}**")
                    
                    # Additional insights
                    with st.expander("📊 Prediction Insights"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Level", "Medium" if premium_amount < 2000 else "High")
                        with col2:
                            monthly_premium = premium_amount / 12
                            st.metric("Monthly Premium", f"${monthly_premium:.2f}")
                        with col3:
                            st.metric("Age Factor", f"{age} years")
                        
                        st.info("💡 **Tip:** Factors like smoking status, vehicle age, and previous claims significantly impact your premium.")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please check that your model is compatible with the input features.")

if __name__ == "__main__":
    main()