import streamlit as st
import numpy as np
import catboost
import pickle

# Load trained CatBoost model
with open("catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature averages (precomputed from dataset)
average_features = {
    "balance_delta": 500,
    "credit_inflow": 2000,
    "debit_outflows": 1500,
    "paycheck": 3000,
    "gambling_amount": 0,
    # Add more features as needed
}

st.title("Loan Default Probability Estimator")

st.sidebar.header("User Input Features")

# User input sliders
user_inputs = {
    "balance_delta": st.sidebar.slider("Balance Delta (Last Year)", -5000, 5000, 0, step=100),
    "credit_inflow": st.sidebar.slider("Credit Inflow", 0, 10000, 2000, step=500),
    "debit_outflows": st.sidebar.slider("Debit Outflows", 0, 10000, 1500, step=500),
    "paycheck": st.sidebar.slider("Paycheck Amount", 0, 10000, 3000, step=500),
    "gambling_amount": st.sidebar.slider("Gambling Amount", 0, 5000, 0, step=100),
}

# Prepare model input
model_input = np.array([
    user_inputs.get(feat, average_features[feat]) for feat in average_features
]).reshape(1, -1)

# Predict probability of default
default_proba = model.predict_proba(model_input)[0][1]

st.subheader("Estimated Probability of Default")
st.write(f"{default_proba:.2%}")
