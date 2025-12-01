"""
IPO Risk Prediction Dashboard
Streamlit Application for Interactive IPO Analysis

Authors: Logan Wesselt, Julian Tashjian, Dylan Bollinger
JLD Inc. LLC. Partners
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="IPO Risk Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        with open('models/best_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        with open('models/best_regressor.pkl', 'rb') as f:
            regressor = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        with open('models/all_classification_models.pkl', 'rb') as f:
            all_classifiers = pickle.load(f)
        with open('models/all_regression_models.pkl', 'rb') as f:
            all_regressors = pickle.load(f)

        return classifier, regressor, scaler, feature_columns, metadata, all_classifiers, all_regressors
    except FileNotFoundError:
        st.error("Model files not found. Please run the Jupyter notebook first to train models.")
        return None, None, None, None, None, None, None

@st.cache_data
def load_data():
    """Load test predictions and results"""
    try:
        test_preds = pd.read_csv('data/test_predictions.csv')
        clf_results = pd.read_csv('data/classification_results.csv')
        reg_results = pd.read_csv('data/regression_results.csv')
        strategy_results = pd.read_csv('data/strategy_summary.csv')
        feature_importance = pd.read_csv('data/feature_importance.csv')

        return test_preds, clf_results, reg_results, strategy_results, feature_importance
    except FileNotFoundError:
        st.error("Data files not found. Please run the Jupyter notebook first.")
        return None, None, None, None, None

# Load everything
classifier, regressor, scaler, feature_columns, metadata, all_classifiers, all_regressors = load_models()
test_preds, clf_results, reg_results, strategy_results, feature_importance = load_data()

# Check if data loaded successfully
if classifier is None or test_preds is None:
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Introduction", "Home & IPO Search", "Model Performance", "Investment Strategies", "Feature Analysis", "IPO Sandbox"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard predicts IPO first-day returns and identifies high-risk offerings 
    using machine learning models trained on synthetic IPO data (2010-2024).
    
    **Models Used:**
    - Classification: """ + metadata['best_classifier_name'] + """
    - Regression: """ + metadata['best_regressor_name'] + """
    
    **Created by:** JLD Inc. LLC. Partners
    """
)

# ============================================================================
# PAGE 0: INTRODUCTION
# ============================================================================
if page == "Introduction":
    st.title("IPO Risk Prediction Dashboard")
    st.markdown("### Machine Learning for First-Day Return Prediction")

    st.markdown("---")

    # Project Overview
    st.markdown("## Project Overview")
    st.write("""
    This dashboard demonstrates the application of machine learning techniques to predict 
    Initial Public Offering (IPO) first-day returns and identify high-risk offerings. 
    The project was developed as part of FIN 377 coursework at Lehigh University.
    """)

    # Research Question
    st.markdown("## Research Question")
    st.info("""
    **Can a machine learning model, using only information available before an IPO's first trading day, 
    effectively predict first-day IPO returns and identify "high-risk" IPOs prone to large negative price moves?**
    """)

    # Data Methodology
    st.markdown("## Data Methodology")

    with st.expander("**Synthetic Data Generation**", expanded=True):
        st.write("""
        This project uses **synthetic IPO data** that mimics real-world patterns observed in academic 
        IPO underpricing literature. The synthetic data was generated using the following methodology:
        
        **Why Synthetic Data?**
        - Real IPO data requires expensive subscriptions (WRDS, Renaissance Capital, Bloomberg)
        - Demonstrates methodology that can be applied to real data
        - Allows for controlled experimentation and testing
        - Educational focus on machine learning techniques rather than data acquisition
        
        **Data Generation Process:**
        1. **Sample Size**: 1,200 IPOs spanning 2010-2024
        2. **Feature Generation**: 40+ features based on:
           - Deal structure (offer price, shares, proceeds)
           - Firm characteristics (age, revenue, profitability)
           - Market conditions (VIX, S&P 500 returns, treasury yields)
        3. **Return Generation**: First-day returns calculated using:
           - Base underpricing rate: ~8% (literature average)
           - Industry effects (e.g., Tech IPOs +4%)
           - VC-backing premium (+3%)
           - Market conditions (VIX, momentum)
           - Random noise component (σ = 12%)
        
        **Realistic Patterns Incorporated:**
        - Technology IPOs show higher average returns
        - VC-backed firms experience greater underpricing
        - Hot markets (low VIX) correlate with higher returns
        - Positive price revisions signal strong demand
        - Young firms have higher first-day volatility
        
        **Academic Foundation:**
        Our synthetic data generation is based on findings from:
        - Loughran & Ritter (2004): IPO underpricing trends
        - Hanley (1993): Partial adjustment phenomenon
        - Lowry & Schwert (2002): Market cycles and IPO performance
        """)

    with st.expander("**Model Training Approach**"):
        st.write("""
        **Temporal Split to Avoid Look-Ahead Bias:**
        - Training: 2010-2019 (800 IPOs)
        - Validation: 2020-2021 (160 IPOs)
        - Test: 2022-2024 (240 IPOs)
        
        **Models Trained:**
        - Logistic Regression (baseline)
        - Random Forest
        - XGBoost
        - LightGBM
        - CatBoost
        
        **Feature Engineering:**
        - Log transformations for skewed variables
        - Interaction terms (e.g., Tech × VC-backed)
        - Market condition indicators
        - Industry dummy variables
        """)

    with st.expander("**Evaluation Metrics**"):
        st.write("""
        **Classification (High-Risk Detection):**
        - ROC-AUC Score: Measures discriminatory power
        - Precision: Accuracy of high-risk predictions
        - Recall: Coverage of actual high-risk IPOs
        - F1 Score: Harmonic mean of precision and recall
        
        **Regression (Return Prediction):**
        - RMSE: Root mean squared error in percentage points
        - MAE: Mean absolute error
        - R²: Proportion of variance explained
        
        **Economic Evaluation:**
        - Portfolio return simulations
        - Sharpe ratios (risk-adjusted returns)
        - Strategy comparison with $1M starting capital
        """)

    # Key Results
    st.markdown("## Key Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best Model AUC", f"{metadata['best_classifier_auc']:.3f}")
        st.caption("Classification performance")

    with col2:
        st.metric("Best Model RMSE", f"{metadata['best_regressor_rmse']:.4f}")
        st.caption("Regression accuracy")

    with col3:
        best_strategy_improvement = strategy_results.iloc[strategy_results['Mean Return (%)'].idxmax()]['Mean Return (%)'] - strategy_results.iloc[0]['Mean Return (%)']
        st.metric("Strategy Improvement", f"+{best_strategy_improvement:.2f}%")
        st.caption("Over naive approach")

    # Limitations
    st.markdown("## Limitations & Caveats")
    st.warning("""
    **Important Considerations:**
    1. **Synthetic Data**: Results based on generated data, not actual IPOs
    2. **Simplified Market**: Real markets have additional complexities
    3. **No Transaction Costs**: Strategy returns don't include fees/commissions
    4. **Static Model**: Real deployment would require continuous retraining
    5. **Feature Availability**: Some features may not be available before IPO
    6. **Market Regime Changes**: Models trained on past data may not predict future behavior
    """)

    # Real-World Application
    st.markdown("## Real-World Application")
    st.write("""
    This methodology can be applied to real IPO data by:
    
    1. **Data Sources**: Subscribe to WRDS, Renaissance Capital, or Bloomberg
    2. **Feature Collection**: Gather firm fundamentals, deal terms, market data
    3. **Model Training**: Use the same pipeline with real data
    4. **Deployment**: Integrate into investment decision processes
    5. **Monitoring**: Track model performance and retrain regularly
    
    **Potential Use Cases:**
    - Investment firms screening IPO opportunities
    - Underwriters assessing pricing risk
    - Retail investors making allocation decisions
    - Academic research on IPO market efficiency
    """)

    # Navigate
    st.markdown("---")
    st.markdown("## Explore the Dashboard")
    st.write("Use the sidebar navigation to:")
    st.write("- **Home & IPO Search**: Look up specific IPOs and view predictions")
    st.write("- **Model Performance**: Compare classification and regression models")
    st.write("- **Investment Strategies**: Evaluate ML-based portfolio strategies")
    st.write("- **Feature Analysis**: Understand which factors drive predictions")
    st.write("- **IPO Sandbox**: Create custom IPO scenarios and test predictions")

# ============================================================================
# PAGE 1: HOME & IPO SEARCH
# ============================================================================
elif page == "Home & IPO Search":
    st.title("IPO Risk Prediction Dashboard")
    st.markdown("### Predict First-Day Returns and Identify High-Risk IPOs")
    st.caption("Based on synthetic IPO data (2010-2024) - See Introduction page for methodology")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total IPOs Analyzed",
            value=f"{len(test_preds):,}",
            delta=None
        )

    with col2:
        st.metric(
            label="Model AUC",
            value=f"{metadata['best_classifier_auc']:.3f}",
            delta=f"+{(metadata['best_classifier_auc']-0.5):.3f} vs Random"
        )

    with col3:
        avg_return = test_preds['first_day_return'].mean() * 100
        st.metric(
            label="Avg First-Day Return",
            value=f"{avg_return:.2f}%",
            delta=None
        )

    with col4:
        high_risk_pct = (test_preds['high_risk_ipo'].sum() / len(test_preds)) * 100
        st.metric(
            label="High-Risk IPOs",
            value=f"{high_risk_pct:.1f}%",
            delta=None
        )

    st.markdown("---")

    # IPO Search Tool
    st.markdown("## IPO Search Tool")
    st.markdown("Search for a specific IPO or browse recent offerings")

    col1, col2 = st.columns([2, 1])

    with col1:
        search_option = st.selectbox(
            "Search by:",
            ["Ticker", "Company Name", "Browse Random IPOs"]
        )

    if search_option == "Ticker":
        ticker = st.selectbox("Select Ticker:", sorted(test_preds['ticker'].unique()))
        selected_ipo = test_preds[test_preds['ticker'] == ticker].iloc[0]

    elif search_option == "Company Name":
        company = st.selectbox("Select Company:", sorted(test_preds['company_name'].unique()))
        selected_ipo = test_preds[test_preds['company_name'] == company].iloc[0]

    else:
        sample_size = st.slider("Number of random IPOs to display:", 5, 20, 10)
        st.markdown("### Random Sample of IPOs")
        sample_ipos = test_preds.sample(n=sample_size)

        display_cols = ['ticker', 'company_name', 'industry', 'first_day_return',
                       'predicted_return', 'predicted_high_risk']
        display_df = sample_ipos[display_cols].copy()
        display_df['first_day_return'] = display_df['first_day_return'].apply(lambda x: f"{x*100:.2f}%")
        display_df['predicted_return'] = display_df['predicted_return'].apply(lambda x: f"{x*100:.2f}%")
        display_df['predicted_high_risk'] = display_df['predicted_high_risk'].map({0: 'Low Risk', 1: 'High Risk'})

        st.dataframe(display_df, use_container_width=True, height=400)

        selected_ticker = st.selectbox("Select an IPO for detailed analysis:", sample_ipos['ticker'].values)
        selected_ipo = test_preds[test_preds['ticker'] == selected_ticker].iloc[0]

    # Display selected IPO details
    if search_option != "Browse Random IPOs" or 'selected_ipo' in locals():
        st.markdown("---")
        st.markdown("## IPO Details")

        # Main IPO card
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"### {selected_ipo['company_name']}")
            st.markdown(f"**Ticker:** {selected_ipo['ticker']}")
            st.markdown(f"**Industry:** {selected_ipo['industry']}")
            st.markdown(f"**IPO Date:** {selected_ipo['ipo_date']}")

        with col2:
            st.markdown("### Actual Performance")
            actual_return = selected_ipo['first_day_return'] * 100
            actual_color = "green" if actual_return >= 0 else "red"
            st.markdown(f"<h2 style='color: {actual_color};'>{actual_return:+.2f}%</h2>", unsafe_allow_html=True)
            st.markdown("First-Day Return")

        with col3:
            st.markdown("### Predicted Return")
            pred_return = selected_ipo['predicted_return'] * 100
            pred_color = "green" if pred_return >= 0 else "red"
            st.markdown(f"<h2 style='color: {pred_color};'>{pred_return:+.2f}%</h2>", unsafe_allow_html=True)
            st.markdown("Model Prediction")

        # Risk assessment
        st.markdown("---")
        st.markdown("### Risk Assessment")

        col1, col2, col3 = st.columns(3)

        risk_prob = selected_ipo['predicted_risk_prob'] * 100
        is_high_risk = selected_ipo['predicted_high_risk'] == 1

        with col1:
            if is_high_risk:
                st.error("**HIGH RISK**")
                st.markdown(f"Risk Probability: **{risk_prob:.1f}%**")
            else:
                st.success("**LOW RISK**")
                st.markdown(f"Risk Probability: **{risk_prob:.1f}%**")

        with col2:
            confidence_level = "High" if abs(risk_prob - 50) > 30 else "Medium" if abs(risk_prob - 50) > 15 else "Low"
            st.metric("Model Confidence", confidence_level)

        with col3:
            was_actually_high_risk = selected_ipo['high_risk_ipo'] == 1
            if is_high_risk == was_actually_high_risk:
                st.success("Correct Prediction")
            else:
                st.warning("Incorrect Prediction")

        # IPO Characteristics
        st.markdown("---")
        st.markdown("### IPO Characteristics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Offer Price", f"${selected_ipo['offer_price']:.2f}")
            st.metric("Firm Age", f"{selected_ipo['firm_age']:.1f} years")

        with col2:
            st.metric("Gross Proceeds", f"${selected_ipo['gross_proceeds']/1e6:.1f}M")
            st.metric("VC-Backed", "Yes" if selected_ipo['vc_backed'] == 1 else "No")

        with col3:
            st.metric("Underwriter Rank", f"{selected_ipo['underwriter_rank']:.0f}/10")
            st.metric("Profitable", "Yes" if selected_ipo['is_profitable'] == 1 else "No")

        with col4:
            st.metric("VIX Level", f"{selected_ipo['vix_level']:.1f}")
            st.metric("Market Momentum", f"{selected_ipo['sp500_1m_return']*100:+.2f}%")

# ============================================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================================
elif page == "Model Performance":
    st.title("Model Performance Analysis")

    # Tabs for classification and regression
    tab1, tab2 = st.tabs(["Classification Models", "Regression Models"])

    with tab1:
        st.markdown("## Classification Performance (High-Risk IPO Detection)")

        # Metrics comparison
        col1, col2 = st.columns(2)

        with col1:
            # AUC Comparison
            fig = go.Figure(data=[
                go.Bar(
                    x=clf_results['Test_AUC'],
                    y=clf_results['Model'],
                    orientation='h',
                    marker=dict(color='steelblue'),
                    text=clf_results['Test_AUC'].apply(lambda x: f'{x:.3f}'),
                    textposition='auto',
                )
            ])
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Random")
            fig.update_layout(
                title="ROC-AUC Score Comparison",
                xaxis_title="AUC Score",
                yaxis_title="Model",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # F1 Score Comparison
            fig = go.Figure(data=[
                go.Bar(
                    x=clf_results['Test_F1'],
                    y=clf_results['Model'],
                    orientation='h',
                    marker=dict(color='coral'),
                    text=clf_results['Test_F1'].apply(lambda x: f'{x:.3f}'),
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="F1 Score Comparison",
                xaxis_title="F1 Score",
                yaxis_title="Model",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics table
        st.markdown("### Detailed Performance Metrics")
        st.dataframe(clf_results, use_container_width=True)

        # Best model highlight
        best_model = clf_results.loc[clf_results['Test_AUC'].idxmax()]
        st.success(f"**Best Model:** {best_model['Model']} with AUC = {best_model['Test_AUC']:.4f}")

        # ROC Curves note
        st.info("ROC curves show the trade-off between True Positive Rate and False Positive Rate. Models closer to the top-left corner perform better.")

    with tab2:
        st.markdown("## Regression Performance (First-Day Return Prediction)")

        # Metrics comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            # RMSE Comparison
            fig = go.Figure(data=[
                go.Bar(
                    x=reg_results['Test_RMSE'],
                    y=reg_results['Model'],
                    orientation='h',
                    marker=dict(color='steelblue'),
                    text=reg_results['Test_RMSE'].apply(lambda x: f'{x:.4f}'),
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="RMSE (Lower is Better)",
                xaxis_title="RMSE",
                yaxis_title="Model",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # MAE Comparison
            fig = go.Figure(data=[
                go.Bar(
                    x=reg_results['Test_MAE'],
                    y=reg_results['Model'],
                    orientation='h',
                    marker=dict(color='coral'),
                    text=reg_results['Test_MAE'].apply(lambda x: f'{x:.4f}'),
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="MAE (Lower is Better)",
                xaxis_title="MAE",
                yaxis_title="Model",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            # R² Comparison
            fig = go.Figure(data=[
                go.Bar(
                    x=reg_results['Test_R2'],
                    y=reg_results['Model'],
                    orientation='h',
                    marker=dict(color='seagreen'),
                    text=reg_results['Test_R2'].apply(lambda x: f'{x:.4f}'),
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="R² Score (Higher is Better)",
                xaxis_title="R² Score",
                yaxis_title="Model",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics table
        st.markdown("### Detailed Performance Metrics")
        st.dataframe(reg_results, use_container_width=True)

        # Best model highlight
        best_model = reg_results.loc[reg_results['Test_RMSE'].idxmin()]
        st.success(f"**Best Model:** {best_model['Model']} with RMSE = {best_model['Test_RMSE']:.4f}")

        # Predicted vs Actual scatter
        st.markdown("### Predicted vs Actual Returns")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_preds['first_day_return'],
            y=test_preds['predicted_return'],
            mode='markers',
            marker=dict(size=5, opacity=0.5, color='steelblue'),
            name='Predictions',
            hovertemplate='<b>Actual:</b> %{x:.2%}<br><b>Predicted:</b> %{y:.2%}<extra></extra>'
        ))

        # Perfect prediction line
        min_val = min(test_preds['first_day_return'].min(), test_preds['predicted_return'].min())
        max_val = max(test_preds['first_day_return'].max(), test_preds['predicted_return'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction'
        ))

        fig.update_layout(
            xaxis_title="Actual First-Day Return",
            yaxis_title="Predicted First-Day Return",
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: INVESTMENT STRATEGIES
# ============================================================================
elif page == "Investment Strategies":
    st.title("Investment Strategies")
    st.markdown("### Comparing ML-Based Investment Approaches")

    # Strategy overview
    col1, col2, col3, col4 = st.columns(4)

    starting_capital = 1_000_000  # $1 million starting capital

    for idx, row in strategy_results.iterrows():
        with [col1, col2, col3, col4][idx]:
            delta_color = "normal" if idx == 0 else "off"
            improvement = row['Mean Return (%)'] - strategy_results.iloc[0]['Mean Return (%)']
            final_value = starting_capital * (1 + row['Total Return (%)'] / 100)

            st.metric(
                label=row['Strategy'],
                value=f"${final_value:,.0f}",
                delta=f"+{improvement:.2f}%" if idx > 0 else f"{row['Mean Return (%)']:.2f}%",
                delta_color=delta_color
            )
            st.caption(f"Return: {row['Mean Return (%)']:.2f}%")

    st.markdown("---")

    # Strategy comparison chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Mean Returns by Strategy")
        fig = go.Figure(data=[
            go.Bar(
                x=strategy_results['Strategy'],
                y=strategy_results['Mean Return (%)'],
                marker=dict(color=['gray', 'steelblue', 'coral', 'seagreen']),
                text=strategy_results['Mean Return (%)'].apply(lambda x: f'{x:.2f}%'),
                textposition='auto',
            )
        ])
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title="Strategy",
            yaxis_title="Mean Return (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Risk-Return Profile")
        fig = go.Figure()

        colors = ['gray', 'steelblue', 'coral', 'seagreen']
        for idx, row in strategy_results.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Std Dev (%)']],
                y=[row['Mean Return (%)']],
                mode='markers+text',
                marker=dict(size=20, color=colors[idx]),
                text=[row['Strategy']],
                textposition='top center',
                name=row['Strategy'],
                showlegend=False
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.3)
        fig.update_layout(
            xaxis_title="Standard Deviation (%)",
            yaxis_title="Mean Return (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed strategy table
    st.markdown("### Strategy Performance Details")
    st.markdown(f"**Starting Capital:** ${starting_capital:,}")

    display_strategy = strategy_results.copy()

    # Add calculated final values and profit/loss
    display_strategy['Final Value ($)'] = display_strategy['Total Return (%)'].apply(
        lambda x: starting_capital * (1 + x / 100)
    )
    display_strategy['Profit/Loss ($)'] = display_strategy['Final Value ($)'] - starting_capital

    # Format for display
    display_strategy['Mean Return (%)'] = display_strategy['Mean Return (%)'].apply(lambda x: f"{x:.2f}%")
    display_strategy['Std Dev (%)'] = display_strategy['Std Dev (%)'].apply(lambda x: f"{x:.2f}%")
    display_strategy['Sharpe Ratio'] = display_strategy['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
    display_strategy['Total Return (%)'] = display_strategy['Total Return (%)'].apply(lambda x: f"{x:.2f}%")
    display_strategy['Final Value ($)'] = display_strategy['Final Value ($)'].apply(lambda x: f"${x:,.2f}")
    display_strategy['Profit/Loss ($)'] = display_strategy['Profit/Loss ($)'].apply(lambda x: f"${x:,.2f}")

    st.dataframe(display_strategy, use_container_width=True)

    # Strategy descriptions
    st.markdown("---")
    st.markdown("### Strategy Descriptions")

    with st.expander("**Naive (All IPOs)**"):
        st.write("""
        - **Approach**: Invest equally in all IPOs
        - **Pros**: Simple, no selection bias
        - **Cons**: Includes high-risk IPOs that may underperform
        """)

    with st.expander("**Avoid High-Risk**"):
        st.write("""
        - **Approach**: Skip IPOs predicted to have < -5% first-day return
        - **Pros**: Filters out potentially problematic IPOs
        - **Cons**: May miss some opportunities
        """)

    with st.expander("**Top Quartile**"):
        st.write("""
        - **Approach**: Only invest in top 25% of predicted returns
        - **Pros**: Focus on best opportunities
        - **Cons**: More concentrated, higher risk
        """)

    with st.expander("**Combined**"):
        st.write("""
        - **Approach**: Top quartile returns AND avoid high-risk
        - **Pros**: Best of both strategies
        - **Cons**: Very selective, may have fewer opportunities
        """)

    # Best strategy recommendation
    best_strategy = strategy_results.loc[strategy_results['Mean Return (%)'].idxmax()]
    improvement = best_strategy['Mean Return (%)'] - strategy_results.iloc[0]['Mean Return (%)']
    best_final_value = starting_capital * (1 + best_strategy['Total Return (%)'] / 100)
    naive_final_value = starting_capital * (1 + strategy_results.iloc[0]['Total Return (%)'] / 100)
    profit_difference = best_final_value - naive_final_value

    st.success(f"""
    **Recommended Strategy:** {best_strategy['Strategy']}
    
    - **Starting Capital:** ${starting_capital:,}
    - **Final Portfolio Value:** ${best_final_value:,.2f}
    - **Total Profit:** ${(best_final_value - starting_capital):,.2f}
    - **Mean Return:** {best_strategy['Mean Return (%)']:.2f}%
    - **Improvement over Naive:** +{improvement:.2f}%
    - **Additional Profit vs Naive:** ${profit_difference:,.2f}
    - **Sharpe Ratio:** {best_strategy['Sharpe Ratio']:.3f}
    - **IPOs Invested:** {int(best_strategy['IPOs Invested'])}
    """)

# ============================================================================
# PAGE 4: FEATURE ANALYSIS
# ============================================================================
elif page == "Feature Analysis":
    st.title("Feature Importance Analysis")
    st.markdown("### Understanding What Drives IPO Performance")

    # Top features
    st.markdown("## Top 10 Most Important Features")

    top_10 = feature_importance.head(10)

    fig = go.Figure(data=[
        go.Bar(
            y=top_10['Feature'],
            x=top_10['Importance'],
            orientation='h',
            marker=dict(
                color=top_10['Importance'],
                colorscale='Viridis',
                showscale=True
            ),
            text=top_10['Importance'].apply(lambda x: f'{x:.4f}'),
            textposition='auto',
        )
    ])
    fig.update_layout(
        xaxis_title="SHAP Importance Value",
        yaxis_title="Feature",
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance table
    st.markdown("### Full Feature Importance Rankings")
    display_importance = feature_importance.copy()
    display_importance['Rank'] = range(1, len(display_importance) + 1)
    display_importance = display_importance[['Rank', 'Feature', 'Importance']]
    display_importance['Importance'] = display_importance['Importance'].apply(lambda x: f'{x:.6f}')

    st.dataframe(display_importance, use_container_width=True, height=400)

    # Key insights
    st.markdown("---")
    st.markdown("## Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top 3 Most Predictive Features")
        for idx in range(min(3, len(feature_importance))):
            feature = feature_importance.iloc[idx]
            st.info(f"""
            **{idx+1}. {feature['Feature']}**
            
            SHAP Importance: {feature['Importance']:.6f}
            """)

    with col2:
        st.markdown("### Feature Categories")

        # Categorize features
        market_features = [f for f in feature_importance['Feature'].values if any(x in f.lower() for x in ['vix', 'sp500', 'treasury', 'market', 'momentum'])]
        deal_features = [f for f in feature_importance['Feature'].values if any(x in f.lower() for x in ['offer', 'price', 'proceeds', 'shares', 'underwriter'])]
        firm_features = [f for f in feature_importance['Feature'].values if any(x in f.lower() for x in ['age', 'revenue', 'assets', 'income', 'profitable'])]

        category_counts = {
            'Market Conditions': len(market_features),
            'Deal Structure': len(deal_features),
            'Firm Characteristics': len(firm_features),
            'Other': len(feature_importance) - len(market_features) - len(deal_features) - len(firm_features)
        }

        fig = go.Figure(data=[
            go.Pie(
                labels=list(category_counts.keys()),
                values=list(category_counts.values()),
                hole=0.4
            )
        ])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Download options
    st.markdown("---")
    st.markdown("## Download Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = test_preds.to_csv(index=False)
        st.download_button(
            label="Download Test Predictions",
            data=csv,
            file_name="ipo_test_predictions.csv",
            mime="text/csv"
        )

    with col2:
        csv = feature_importance.to_csv(index=False)
        st.download_button(
            label="Download Feature Importance",
            data=csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )

    with col3:
        csv = strategy_results.to_csv(index=False)
        st.download_button(
            label="Download Strategy Results",
            data=csv,
            file_name="strategy_results.csv",
            mime="text/csv"
        )

# ============================================================================
# PAGE 6: IPO SANDBOX
# ============================================================================
elif page == "IPO Sandbox":
    st.title("IPO Sandbox")
    st.markdown("### Create Your Own IPO Scenario")

    st.write("""
    Use the interactive controls below to create a custom IPO scenario based on the top 10 most 
    influential factors identified by our models. The dashboard will predict the expected 
    first-day return and risk classification.
    """)

    st.markdown("---")

    # Get top 10 features
    top_10_features = feature_importance.head(10)['Feature'].tolist()

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    # Dictionary to store user inputs
    user_inputs = {}

    # Common IPO parameters - left column
    with col1:
        st.markdown("### Deal Structure")

        user_inputs['offer_price'] = st.slider(
            "Offer Price ($)",
            min_value=10.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="The price per share at which the IPO is offered"
        )

        user_inputs['shares_offered'] = st.number_input(
            "Shares Offered (millions)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Number of shares being offered in millions"
        ) * 1e6

        user_inputs['pct_primary'] = st.slider(
            "% Primary Shares",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Percentage of new shares vs. existing shares being sold"
        )

        user_inputs['price_range_deviation'] = st.slider(
            "Price Range Deviation",
            min_value=-0.3,
            max_value=0.3,
            value=0.0,
            step=0.05,
            help="Deviation from initial filing price range (positive = priced above range)"
        )

        st.markdown("### Firm Characteristics")

        user_inputs['firm_age'] = st.slider(
            "Firm Age (years)",
            min_value=1,
            max_value=50,
            value=8,
            step=1,
            help="Years since company founding"
        )

        user_inputs['revenue'] = st.number_input(
            "Annual Revenue ($M)",
            min_value=1.0,
            max_value=10000.0,
            value=500.0,
            step=10.0,
            help="Annual revenue in millions of dollars"
        ) * 1e6

        user_inputs['total_assets'] = st.number_input(
            "Total Assets ($M)",
            min_value=10.0,
            max_value=50000.0,
            value=1000.0,
            step=50.0,
            help="Total assets in millions of dollars"
        ) * 1e6

        user_inputs['is_profitable'] = st.selectbox(
            "Is Profitable?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=1,
            help="Whether the company is currently profitable"
        )

    # Market conditions and quality - right column
    with col2:
        st.markdown("### Deal Quality")

        user_inputs['vc_backed'] = st.selectbox(
            "VC-Backed?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=1,
            help="Whether the company has venture capital backing"
        )

        user_inputs['underwriter_rank'] = st.slider(
            "Underwriter Rank",
            min_value=1,
            max_value=10,
            value=7,
            step=1,
            help="Quality ranking of lead underwriter (10 = highest)"
        )

        user_inputs['industry'] = st.selectbox(
            "Industry",
            options=['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial',
                    'Energy', 'Materials', 'Utilities', 'Real Estate', 'Communication'],
            index=0,
            help="Primary industry classification"
        )

        st.markdown("### Market Conditions")

        user_inputs['vix_level'] = st.slider(
            "VIX Level",
            min_value=10.0,
            max_value=40.0,
            value=18.0,
            step=1.0,
            help="CBOE Volatility Index at time of IPO"
        )

        user_inputs['sp500_1m_return'] = st.slider(
            "S&P 500 1-Month Return (%)",
            min_value=-10.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="S&P 500 return over the month prior to IPO"
        ) / 100

        user_inputs['sp500_3m_return'] = st.slider(
            "S&P 500 3-Month Return (%)",
            min_value=-20.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            help="S&P 500 return over three months prior to IPO"
        ) / 100

        user_inputs['treasury_10y'] = st.slider(
            "10-Year Treasury Yield (%)",
            min_value=1.0,
            max_value=6.0,
            value=3.5,
            step=0.1,
            help="10-year U.S. Treasury yield at time of IPO"
        )

        user_inputs['market_volatility'] = st.slider(
            "Market Volatility",
            min_value=0.05,
            max_value=0.40,
            value=0.15,
            step=0.01,
            help="General market volatility measure"
        )

    st.markdown("---")

    # Calculate derived features
    user_inputs['gross_proceeds'] = user_inputs['offer_price'] * user_inputs['shares_offered']
    user_inputs['log_proceeds'] = np.log(user_inputs['gross_proceeds'])
    user_inputs['implied_valuation'] = user_inputs['gross_proceeds'] / user_inputs['pct_primary']
    user_inputs['proceeds_per_share'] = user_inputs['gross_proceeds'] / user_inputs['shares_offered']
    user_inputs['log_assets'] = np.log(user_inputs['total_assets'])
    user_inputs['log_revenue'] = np.log(user_inputs['revenue'] + 1)
    user_inputs['net_income'] = user_inputs['revenue'] * 0.1 if user_inputs['is_profitable'] else -user_inputs['revenue'] * 0.05
    user_inputs['valuation_to_assets'] = user_inputs['implied_valuation'] / user_inputs['total_assets']
    user_inputs['revenue_to_assets'] = user_inputs['revenue'] / user_inputs['total_assets']
    user_inputs['pct_secondary'] = 1 - user_inputs['pct_primary']
    user_inputs['lockup_period'] = 180  # default

    # Binary indicators
    user_inputs['is_young_firm'] = 1 if user_inputs['firm_age'] < 5 else 0
    user_inputs['is_tech'] = 1 if user_inputs['industry'] == 'Technology' else 0
    user_inputs['high_vix'] = 1 if user_inputs['vix_level'] > 20 else 0
    user_inputs['positive_momentum'] = 1 if user_inputs['sp500_1m_return'] > 0 else 0

    # Interactions
    user_inputs['tech_x_vc'] = user_inputs['is_tech'] * user_inputs['vc_backed']
    user_inputs['young_x_vc'] = user_inputs['is_young_firm'] * user_inputs['vc_backed']
    user_inputs['vix_x_momentum'] = user_inputs['vix_level'] * user_inputs['sp500_1m_return']

    # Industry dummies
    for ind in ['Healthcare', 'Financial', 'Consumer', 'Industrial', 'Energy', 'Materials', 'Utilities', 'Real Estate', 'Communication']:
        user_inputs[f'industry_{ind}'] = 1 if user_inputs['industry'] == ind else 0

    # Prepare feature vector
    feature_vector = []
    for col in feature_columns:
        if col in user_inputs:
            feature_vector.append(user_inputs[col])
        else:
            feature_vector.append(0)  # Default for missing features

    feature_vector = np.array(feature_vector).reshape(1, -1)

    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)

    # Make predictions
    if st.button("Predict IPO Performance", type="primary"):
        st.markdown("---")
        st.markdown("## Prediction Results")

        # Get predictions
        risk_prob = classifier.predict_proba(feature_vector_scaled)[0][1]
        predicted_return = regressor.predict(feature_vector_scaled)[0]
        is_high_risk = risk_prob >= 0.5

        # Display results in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Risk Classification")
            if is_high_risk:
                st.error("**HIGH RISK**")
            else:
                st.success("**LOW RISK**")
            st.metric("Risk Probability", f"{risk_prob*100:.1f}%")

        with col2:
            st.markdown("### Predicted Return")
            return_color = "green" if predicted_return >= 0 else "red"
            st.markdown(f"<h2 style='color: {return_color};'>{predicted_return*100:+.2f}%</h2>", unsafe_allow_html=True)
            st.caption("Expected first-day return")

        with col3:
            st.markdown("### Confidence Level")
            confidence = "High" if abs(risk_prob - 0.5) > 0.3 else "Medium" if abs(risk_prob - 0.5) > 0.15 else "Low"
            st.metric("Model Confidence", confidence)
            st.caption("Based on probability distance from 50%")

        # Investment recommendation
        st.markdown("---")
        st.markdown("### Investment Recommendation")

        if not is_high_risk and predicted_return > 0.10:
            st.success("""
            **Strong Buy Candidate**
            - Low risk of negative returns
            - Predicted return exceeds 10%
            - Favorable market conditions
            """)
        elif not is_high_risk and predicted_return > 0.05:
            st.info("""
            **Moderate Buy Candidate**
            - Low risk classification
            - Positive expected returns
            - Consider portfolio allocation limits
            """)
        elif is_high_risk and predicted_return > 0:
            st.warning("""
            **Cautious Approach Recommended**
            - High risk classification despite positive expected return
            - Higher volatility expected
            - Consider smaller position size
            """)
        else:
            st.error("""
            **Avoid**
            - High risk of negative returns
            - Unfavorable risk-return profile
            - Consider waiting for better opportunities
            """)

        # Feature contributions
        st.markdown("---")
        st.markdown("### Key Factors Influencing This Prediction")

        st.write("**Most Influential Features for This IPO:**")

        # Display user's values for top features
        important_factors = []
        for feature in top_10_features[:5]:
            if feature in user_inputs:
                value = user_inputs[feature]
                if isinstance(value, float):
                    if abs(value) < 1:
                        important_factors.append(f"- **{feature}**: {value:.3f}")
                    elif abs(value) > 1000000:
                        important_factors.append(f"- **{feature}**: ${value/1e6:.1f}M")
                    else:
                        important_factors.append(f"- **{feature}**: {value:.2f}")
                else:
                    important_factors.append(f"- **{feature}**: {value}")

        for factor in important_factors:
            st.write(factor)

        # Scenario comparison
        st.markdown("---")
        st.markdown("### Compare with Averages")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Your IPO Scenario:**")
            st.write(f"- Offer Price: ${user_inputs['offer_price']:.2f}")
            st.write(f"- VC-Backed: {'Yes' if user_inputs['vc_backed'] else 'No'}")
            st.write(f"- Industry: {user_inputs['industry']}")
            st.write(f"- VIX Level: {user_inputs['vix_level']:.1f}")
            st.write(f"- Firm Age: {user_inputs['firm_age']} years")

        with col2:
            st.write("**Dataset Averages:**")
            st.write(f"- Avg Offer Price: ${test_preds['offer_price'].mean():.2f}")
            st.write(f"- VC-Backed Rate: {test_preds['vc_backed'].mean()*100:.0f}%")
            st.write(f"- Avg VIX: {test_preds['vix_level'].mean():.1f}")
            st.write(f"- Avg Firm Age: {test_preds['firm_age'].mean():.1f} years")
            st.write(f"- Avg Return: {test_preds['first_day_return'].mean()*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>IPO Risk Prediction Dashboard</strong></p>
    <p>Created by Logan Wesselt, Julian Tashjian, Dylan Bollinger | JLD Inc. LLC. Partners</p>
    <p>FIN 377 Final Project | Machine Learning for IPO Analysis</p>
</div>
""", unsafe_allow_html=True)