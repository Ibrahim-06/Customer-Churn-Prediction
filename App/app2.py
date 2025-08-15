import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== Load Models & Data =====================
model = joblib.load("../Model/Logistic_model.pkl")
scaler = joblib.load("../Scaler/Scaler.pkl")
feature_cols = joblib.load("../Features/features.pkl")

# ===================== Page Config =====================
st.set_page_config(page_title="Churn Predictor", layout="wide")

# ===================== Custom CSS =====================
st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}

body {
    background: linear-gradient(135deg, #1d2b64, #f8cdda);
    color: white !important;
}

h1 {
    text-align: center;
    font-size: 3.5em;
    font-weight: bold;
    color: #fff;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.4);
}

.stSelectbox, .stSlider, .stTextInput {
    background-color: rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    color: white !important;
}
.stSelectbox:hover, .stSlider:hover, .stTextInput:hover {
    background-color: rgba(255,255,255,0.2) !important;
}

.stButton>button {
    background: linear-gradient(45deg, #ff4b2b, #ff416c);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.7em 2em;
    font-size: 1.2em;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(45deg, #ff416c, #ff4b2b);
}

.result-box {
    background-color: rgba(255,255,255,0.12);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===================== Title =====================
st.markdown("<h1>🔮 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Fill in the details below to check churn probability")

# ===================== Inputs =====================
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

with col3:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.markdown("### 💵 Billing Details")
colA, colB, colC = st.columns(3)
with colA:
    tenure = st.slider("📅 Tenure (Months)", 0, 72, 12)
with colB:
    MonthlyCharges = st.slider("💰 Monthly Charges", 0.0, 120.0, 70.0)
with colC:
    TotalCharges = st.slider("💵 Total Charges", 0.0, 10000.0, 350.0)

# ===================== Data Processing =====================
data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([data])
input_df_encoded = pd.get_dummies(input_df)
for col in feature_cols:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0
input_df_encoded = input_df_encoded[feature_cols]

# ===================== Prediction =====================
if st.button("🚀 Predict Now"):
    input_scaled = scaler.transform(input_df_encoded)
    proba = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if proba >= 0.28 else 0

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if prediction == 1:
        st.error(f"⚠️ Likely to churn! Probability: {proba*100:.2f}%")
    else:
        st.success(f"✅ Not likely to churn. Probability: {proba*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # ===================== Heatmap =====================
    st.markdown("### 🔥 Feature Heatmap for This Customer")
    heat_data = pd.DataFrame(input_df_encoded.iloc[0]).T
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap(heat_data, cmap="coolwarm", annot=False, cbar=True, ax=ax)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
