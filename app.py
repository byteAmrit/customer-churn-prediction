import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/churn_model.pkl")

st.success("✅ Model Loaded Successfully")

# ---------------- TITLE ----------------
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Predict whether a customer will churn based on behavior data.")

st.divider()

# ---------------- INPUT SECTION ----------------
st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    usage_frequency = st.slider("Usage Frequency", 0, 50, 10)
    support_calls = st.slider("Support Calls", 0, 20, 2)
    payment_delay = st.slider("Payment Delay (Days)", 0, 30, 5)

st.divider()

# ---------------- SUBSCRIPTION ----------------
st.header("📦 Subscription Details")

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

contract_length = st.slider("Contract Length (Months)", 0, 24, 12)
total_spend = st.number_input("Total Spend", 0, 10000, 1000)
last_interaction = st.slider("Days Since Last Interaction", 0, 60, 10)

# ---------------- ENCODING ----------------
gender_map = {"Male": 1, "Female": 0}
sub_map = {"Basic": 0, "Standard": 1, "Premium": 2}

gender = gender_map[gender]
subscription_type = sub_map[subscription_type]

# ---------------- FEATURE ORDER (VERY IMPORTANT) ----------------
input_data = np.array([[
    age,
    gender,
    tenure,
    usage_frequency,
    support_calls,
    payment_delay,
    subscription_type,
    contract_length,
    total_spend,
    last_interaction
]])

# ---------------- PREDICTION ----------------
st.divider()

if st.button("🚀 Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer Likely to Churn (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Customer Likely to Stay (Risk: {probability*100:.2f}%)")
