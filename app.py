import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import os

# =====================================================================================
# Load Models, Data, Metadata
# =====================================================================================

@st.cache_resource
def load_artifacts():
    # Load models
    with open("models/best_classifier.pkl", "rb") as f:
        best_clf = pickle.load(f)
    with open("models/best_regressor.pkl", "rb") as f:
        best_reg = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    # Load metadata
    with open("models/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Load data outputs
    test_predictions = pd.read_csv("data/test_predictions.csv")
    clf_results = pd.read_csv("data/classification_results.csv")
    reg_results = pd.read_csv("data/regression_results.csv")
    strategy_summary = pd.read_csv("data/strategy_summary.csv")
    importance_df = pd.read_csv("data/feature_importance.csv")

    return {
        "clf": best_clf,
        "reg": best_reg,
        "scaler": scaler,
        "features": feature_cols,
        "metadata": metadata,
        "preds": test_predictions,
        "clf_results": clf_results,
        "reg_results": reg_results,
        "strategy": strategy_summary,
        "importance": importance_df
    }

artifacts = load_artifacts()

clf_model = artifacts["clf"]
reg_model = artifacts["reg"]
scaler = artifacts["scaler"]
feature_cols = artifacts["features"]
metadata = artifacts["metadata"]
test_predictions = artifacts["preds"]
clf_results = artifacts["clf_results"]
reg_results = artifacts["reg_results"]
strategy_summary = artifacts["strategy"]
importance_df = artifacts["importance"]


# =====================================================================================
# Streamlit Setup
# =====================================================================================

st.set_page_config(
    page_title="IPO Risk & First-Day Return Prediction",
    layout="wide",
)

st.title("IPO Risk & First-Day Performance Dashboard")
st.write("Built by JLD Inc. LLC. Partners – FIN 377 Final Project")


# =====================================================================================
# Utility: Format Prediction Output
# =====================================================================================

def predict_from_input(input_df):
    scaled = scaler.transform(input_df[feature_cols])
    risk_prob = clf_model.predict_proba(scaled)[:, 1][0]
    predicted_return = reg_model.predict(scaled)[0]
    return risk_prob, predicted_return


def risk_category(prob):
    if prob >= 0.50:
        return "High Risk (↓ Expected Negative Return)"
    elif prob >= 0.30:
        return "Moderate Risk"
    else:
        return "Low Risk"


# =====================================================================================
# Sidebar Navigation
# =====================================================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "IPO Search",
        "Visual Analytics",
        "Model Performance",
        "Investment Strategies"
    ]
)


# =====================================================================================
# PAGE 1 — IPO Search Tool
# =====================================================================================

if page == "IPO Search":

    st.header("IPO Search Tool")

    st.write("Search an IPO included in the model's test set.")

    ticker_list = sorted(test_predictions["ticker"].unique())
    selected = st.selectbox("Select an IPO Ticker", ticker_list)

    row = test_predictions[test_predictions["ticker"] == selected].iloc[0]

    st.subheader(f"Company: {row['company_name']} ({row['ticker']})")

    st.write("### Model Outputs")

    risk = row["predicted_risk_prob"]
    pred_return = row["predicted_return"]
    risk_flag = row["predicted_high_risk"]

    st.metric("Predicted First-Day Return", f"{pred_return*100:.2f}%")
    st.metric("Risk Probability", f"{risk:.2%}")
    st.metric("Risk Classification", risk_category(risk))

    with st.expander("Show Full Input Features"):
        st.dataframe(row[feature_cols])



# =====================================================================================
# PAGE 2 — Visual Analytics
# =====================================================================================

elif page == "Visual Analytics":

    st.header("Visual Analytics")

    col1, col2 = st.columns(2)

    # SHAP Feature Importance Bar
    with col1:
        st.subheader("Top Predictive Features (SHAP Importance)")
        fig, ax = plt.subplots(figsize=(6, 8))
        top10 = importance_df.sort_values("Importance", ascending=False).head(10)

        ax.barh(top10["Feature"], top10["Importance"])
        ax.set_xlabel("Mean |SHAP Value|")
        ax.invert_yaxis()
        st.pyplot(fig)

    # Distribution of predicted returns
    with col2:
        st.subheader("Distribution of Predicted First-Day Returns")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(test_predictions["predicted_return"] * 100, bins=25)
        ax.set_xlabel("Predicted Return (%)")
        st.pyplot(fig)

    st.subheader("VIX vs Predicted IPO Risk")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(test_predictions["vix_level"], test_predictions["predicted_risk_prob"], alpha=0.5)
    ax.set_xlabel("VIX Level")
    ax.set_ylabel("Predicted Risk Probability")
    st.pyplot(fig)



# =====================================================================================
# PAGE 3 — Model Performance
# =====================================================================================

elif page == "Model Performance":

    st.header("Model Performance Overview")

    st.subheader("Classification Models — High-Risk IPO Prediction")
    st.dataframe(clf_results)

    st.subheader("Regression Models — First-Day Return Prediction")
    st.dataframe(reg_results)

    st.subheader("Best Models")
    st.write(f"**Best Classifier:** {metadata['best_classifier_name']}  (AUC = {metadata['best_classifier_auc']:.3f})")
    st.write(f"**Best Regressor:** {metadata['best_regressor_name']}  (RMSE = {metadata['best_regressor_rmse']:.3f})")



# =====================================================================================
# PAGE 4 — Investment Strategies
# =====================================================================================

elif page == "Investment Strategies":

    st.header("Economic Evaluation of IPO Investment Strategies")

    st.write(
        "The following strategy outputs replicate those computed in the research notebook, "
        "based on predicted returns and predicted risk classification."
    )

    st.subheader("Strategy Summary Table")
    st.dataframe(strategy_summary)

    # Bar chart of mean returns
    st.subheader("Mean Return by Strategy")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(strategy_summary["Strategy"], strategy_summary["Mean Return (%)"])
    ax.set_ylabel("Mean Return (%)")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    # Risk-return profile
    st.subheader("Risk–Return Profile")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        strategy_summary["Std Dev (%)"],
        strategy_summary["Mean Return (%)"],
        s=150
    )
    for i, label in enumerate(strategy_summary["Strategy"]):
        ax.annotate(label, (strategy_summary["Std Dev (%)"][i], strategy_summary["Mean Return (%)"][i]))

    ax.set_xlabel("Standard Deviation (%)")
    ax.set_ylabel("Mean Return (%)")
    st.pyplot(fig)
