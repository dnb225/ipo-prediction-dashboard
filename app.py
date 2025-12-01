import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================================================
# Page Configuration and Theme
# =====================================================================================

st.set_page_config(
    page_title="IPO Prediction and Analysis Suite",
    layout="wide",
)

sns.set_style("whitegrid")

st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .sidebar .sidebar-content { background-color: #f0f4ff; }
    h1, h2, h3 { color: #003366; }
</style>
""", unsafe_allow_html=True)


# =====================================================================================
# Load All Artifacts
# =====================================================================================

@st.cache_resource
def load_artifacts():
    with open("models/best_classifier.pkl", "rb") as f:
        best_clf = pickle.load(f)
    with open("models/best_regressor.pkl", "rb") as f:
        best_reg = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("models/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Load analytics
    preds = pd.read_csv("data/test_predictions.csv")
    clf_metrics = pd.read_csv("data/classification_results.csv")
    reg_metrics = pd.read_csv("data/regression_results.csv")
    strategy_summary = pd.read_csv("data/strategy_summary.csv")
    importance = pd.read_csv("data/feature_importance.csv")

    return {
        "clf": best_clf,
        "reg": best_reg,
        "scaler": scaler,
        "features": feature_cols,
        "metadata": metadata,
        "preds": preds,
        "clf_metrics": clf_metrics,
        "reg_metrics": reg_metrics,
        "strategy": strategy_summary,
        "importance": importance
    }


art = load_artifacts()

clf = art["clf"]
reg = art["reg"]
scaler = art["scaler"]
feature_cols = art["features"]
metadata = art["metadata"]
preds = art["preds"]
clf_metrics = art["clf_metrics"]
reg_metrics = art["reg_metrics"]
strategy = art["strategy"]
importance = art["importance"]


# =====================================================================================
# Helper Functions
# =====================================================================================

def risk_category(prob):
    if prob >= 0.50:
        return ("High Risk", "red")
    elif prob >= 0.30:
        return ("Moderate Risk", "orange")
    return ("Low Risk", "green")


def predict(df):
    scaled = scaler.transform(df[feature_cols])
    p_risk = clf.predict_proba(scaled)[0, 1]
    p_ret = reg.predict(scaled)[0]
    return p_risk, p_ret


# =====================================================================================
# Sidebar Navigation
# =====================================================================================

page = st.sidebar.radio(
    "Navigate",
    [
        "IPO Search",
        "Mock IPO Builder",
        "Scenario Stress Testing",
        "Feature Importance",
        "Model Performance",
        "Strategy Evaluation"
    ]
)


# =====================================================================================
# 1. IPO Search Page
# =====================================================================================

if page == "IPO Search":

    st.title("IPO Prediction Search Tool")

    tickers = sorted(preds["ticker"].unique())
    t = st.selectbox("Select an IPO", tickers)

    row = preds[preds["ticker"] == t].iloc[0]

    st.subheader(f"{row['company_name']} ({row['ticker']})")

    with st.container():
        col1, col2, col3 = st.columns(3)

        col1.metric("Predicted First-Day Return", f"{row['predicted_return']*100:.2f}%")
        col2.metric("Risk Probability", f"{row['predicted_risk_prob']:.2%}")
        category, col = risk_category(row['predicted_risk_prob'])
        col3.markdown(f"<h3 style='color:{col}'>{category}</h3>", unsafe_allow_html=True)

    with st.expander("Full Feature Input"):
        st.dataframe(row[feature_cols])


# =====================================================================================
# 2. Mock IPO Builder
# =====================================================================================

elif page == "Mock IPO Builder":
    st.title("Mock IPO Builder and Simulator")

    st.write("Create your own hypothetical IPO and evaluate predicted risk and return.")

    inputs = {}

    with st.container():
        col1, col2 = st.columns(2)

        for i, colname in enumerate(feature_cols):
            if "price" in colname.lower():
                inputs[colname] = col1.slider(colname, 0.0, 200.0, 50.0)
            elif "shares" in colname.lower() or "proceeds" in colname.lower():
                inputs[colname] = col1.slider(colname, 0, 500_000_000, 50_000_000)
            elif "age" in colname.lower():
                inputs[colname] = col2.slider(colname, 0, 100, 10)
            elif "vix" in colname.lower():
                inputs[colname] = col2.slider(colname, 5.0, 60.0, 20.0)
            else:
                inputs[colname] = col2.slider(colname, -5.0, 5.0, 0.0)

    df = pd.DataFrame([inputs])

    p_risk, p_ret = predict(df)
    category, col = risk_category(p_risk)

    st.subheader("Simulation Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted First-Day Return", f"{p_ret*100:.2f}%")
    c2.metric("Risk Probability", f"{p_risk:.2%}")
    c3.markdown(f"<h3 style='color:{col}'>{category}</h3>", unsafe_allow_html=True)


# =====================================================================================
# 3. Scenario Stress Testing
# =====================================================================================

elif page == "Scenario Stress Testing":
    st.title("Market Condition Stress Testing")

    st.write("Evaluate how changes in market volatility, interest rates, or deal characteristics influence risk.")

    ipodf = pd.DataFrame([preds[feature_cols].mean()])  # baseline IPO example

    vix = st.slider("CBOE VIX Level", 5.0, 60.0, 20.0)
    ipodf["vix_level"] = vix

    shares = st.slider("Shares Offered", 0, 300_000_000, int(ipodf["shares_offered"].iloc[0]))
    ipodf["shares_offered"] = shares

    proceeds = st.slider("Gross Proceeds ($)", 1e6, 500e6, float(ipodf["gross_proceeds"].iloc[0]))
    ipodf["gross_proceeds"] = proceeds

    p_risk, p_ret = predict(ipodf)
    category, col = risk_category(p_risk)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted First-Day Return", f"{p_ret*100:.2f}%")
    c2.metric("Risk Probability", f"{p_risk:.2%}")
    c3.markdown(f"<h3 style='color:{col}'>{category}</h3>", unsafe_allow_html=True)

    st.subheader("Effect of VIX on Model Predictions")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(preds["vix_level"], preds["predicted_risk_prob"], alpha=0.6)
    ax.axvline(vix, color="red")
    ax.set_xlabel("VIX")
    ax.set_ylabel("Risk Probability")
    st.pyplot(fig)


# =====================================================================================
# 4. Feature Importance
# =====================================================================================

elif page == "Feature Importance":
    st.title("Feature Importance and Explainability")

    st.subheader("Top Predictive Features (SHAP Importance)")
    fig, ax = plt.subplots(figsize=(6, 8))
    top10 = importance.sort_values("Importance", ascending=False).head(10)
    ax.barh(top10["Feature"], top10["Importance"], color="#0055A4")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.subheader("SHAP Summary Plot")
    explainer = shap.TreeExplainer(clf)
    sample = scaler.transform(preds[feature_cols].head(200))
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, pd.DataFrame(sample, columns=feature_cols), show=False)
    st.pyplot(fig)


# =====================================================================================
# 5. Model Performance
# =====================================================================================

elif page == "Model Performance":
    st.title("Model Performance Overview")

    st.subheader("Classification Results")
    st.dataframe(clf_metrics)

    st.subheader("Regression Results")
    st.dataframe(reg_metrics)

    st.write("Best Classifier:", metadata["best_classifier_name"])
    st.write("Best Regressor:", metadata["best_regressor_name"])


# =====================================================================================
# 6. Strategy Evaluation
# =====================================================================================

elif page == "Strategy Evaluation":
    st.title("Investment Strategy Evaluation")

    st.write("Comparison of hypothetical investment strategies using model output.")

    st.subheader("Summary Table")
    st.dataframe(strategy)

    st.subheader("Mean Returns")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(strategy, x="Strategy", y="Mean Return (%)", ax=ax, palette="Blues_r")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    st.subheader("Risk–Return Scatter")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(strategy["Std Dev (%)"], strategy["Mean Return (%)"], s=120, c="#0066CC")
    for i, row in strategy.iterrows():
        ax.text(row["Std Dev (%)"] + 0.1, row["Mean Return (%)"], row["Strategy"])
    ax.set_xlabel("Std Dev (%)")
    ax.set_ylabel("Mean Return (%)")
    st.pyplot(fig)
