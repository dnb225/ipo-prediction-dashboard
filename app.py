import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------
# Load models and data
# ---------------------------------------------
@st.cache_resource
def load_models():
    with open("best_clf.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("best_reg.pkl", "rb") as f:
        reg = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    return clf, reg, scaler, feature_cols

clf, reg, scaler, feature_cols = load_models()

# Optional dataset (for search + charts)
@st.cache_resource
def load_ipo_data():
    try:
        return pd.read_csv("ipo_data.csv")
    except:
        return None

ipo_df = load_ipo_data()

st.set_page_config(page_title="IPO Risk & Return Dashboard", layout="wide")

# ---------------------------------------------
# Dashboard Header
# ---------------------------------------------
st.title("IPO Risk & First-Day Return Prediction Dashboard")
st.markdown("Built using machine learning models developed in *dnb225_test_2.py*.")

# ------------------------------
# Sidebar – IPO Search Tool
# ------------------------------
st.sidebar.header("IPO Search")

ticker = st.sidebar.text_input("Enter IPO Ticker (e.g., IPO0123)")

st.sidebar.markdown("---")
st.sidebar.write("Or manually input pre-IPO characteristics:")

input_data = {}
for col in feature_cols:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

# Convert dict → DataFrame
user_df = pd.DataFrame([input_data])

# ---------------------------------------------
# Prediction Functions
# ---------------------------------------------
def run_predictions(df):
    scaled = scaler.transform(df)
    pred_prob = clf.predict_proba(scaled)[0, 1]
    pred_return = reg.predict(scaled)[0]
    risk_class = (
        "High Risk" if pred_prob >= 0.50 else
        "Moderate Risk" if pred_prob >= 0.25 else
        "Low Risk"
    )
    return pred_prob, pred_return, risk_class

# ---------------------------------------------------
# IPO Search (if dataset exists)
# ---------------------------------------------------
if ticker and ipo_df is not None:
    if ticker in ipo_df["ticker"].values:
        st.subheader(f"Results for: **{ticker}**")

        row = ipo_df[ipo_df["ticker"] == ticker].iloc[0]
        model_input = row[feature_cols].to_frame().T

        pred_prob, pred_return, risk_class = run_predictions(model_input)

        st.metric("Predicted First-Day Return", f"{pred_return*100:.2f}%")
        st.metric("Predicted High-Risk Probability", f"{pred_prob*100:.2f}%")
        st.metric("Risk Classification", risk_class)

    else:
        st.warning("Ticker not found in dataset.")

# ---------------------------------------------------
# Manual Input Predictions
# ---------------------------------------------------
st.header("Model Predictions")
pred_prob, pred_return, risk_class = run_predictions(user_df)

col1, col2, col3 = st.columns(3)
col1.metric("Predicted First-Day Return", f"{pred_return*100:.2f}%")
col2.metric("High-Risk Probability", f"{pred_prob*100:.2f}%")
col3.metric("Risk Category", risk_class)

# ---------------------------------------------------
# SHAP Feature Importance
# ---------------------------------------------------
st.header("Feature Importance (SHAP)")

explainer = shap.TreeExplainer(clf)
sample_scaled = scaler.transform(user_df)
shap_values = explainer.shap_values(sample_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_values, pd.DataFrame(sample_scaled, columns=feature_cols), show=False)
st.pyplot(fig)

# ---------------------------------------------------
# Dataset Exploration (Optional)
# ---------------------------------------------------
if ipo_df is not None:
    st.header("Dataset Explorer")

    st.write("Full IPO dataset loaded:")
    st.dataframe(ipo_df.head())

    st.subheader("Distribution of First-Day Returns")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(ipo_df["first_day_return"], bins=50)
    ax2.set_xlabel("First-Day Return")
    st.pyplot(fig2)

st.markdown("---")
st.write("© JLD Inc. LLC. Partners – FIN 377 Final Project")