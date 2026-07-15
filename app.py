import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the trained artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, scaler, le

model, scaler, le = load_artifacts()

# 2. UI Layout
st.title("🛒 Sales Demand Spike Predictor")
st.write("Enter product metrics to predict if tomorrow will be a high-demand spike.")

# Define input fields (matching your feature columns)
col1, col2 = st.columns(2)
with col1:
    marketing = st.number_input("Marketing Spend", value=200.0)
    unit_price = st.number_input("Unit Price", value=800.0)
    sentiment = st.number_input("Consumer Sentiment", value=80.0)
with col2:
    is_holiday = st.selectbox("Is Holiday?", [0, 1])
    is_weekend = st.selectbox("Is Weekend?", [0, 1])
    # Add other necessary input fields here...

# 3. Prediction Logic
if st.button("Predict Spike"):
    # Create a feature array (Ensure order matches training!)
    input_data = np.array([[marketing, unit_price, sentiment, is_holiday, is_weekend]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction[0] == 1:
        st.error(f"High Demand Spike Predicted! (Confidence: {probability:.2%})")
    else:
        st.success(f"Normal Demand Expected. (Confidence: {1-probability:.2%})")