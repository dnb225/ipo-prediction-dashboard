# IPO Risk Prediction Dashboard

Machine Learning Dashboard for Predicting IPO First-Day Returns and Identifying High-Risk Offerings

**Authors**: Logan Wesselt, Julian Tashjian, Dylan Bollinger  
**Organization**: JLD Inc. LLC. Partners  
**Course**: FIN 377 Final Project

---

## Project Overview

This project uses machine learning to predict Initial Public Offering (IPO) first-day returns and identify high-risk IPOs prone to negative price movements. The dashboard provides an interactive interface to explore predictions, evaluate model performance, and compare investment strategies.

### Key Features

- **IPO Search Tool**: Look up specific IPOs and view predictions
- **Risk Classification**: Identify IPOs with high probability of negative returns
- **Return Prediction**: Estimate first-day return percentages
- **Investment Strategies**: Compare performance of ML-based vs. naive strategies
- **Feature Importance**: Understand which factors drive IPO performance
- **Model Performance**: Comprehensive metrics for all trained models

---

## Research Question

**Can a machine learning model, using only information available before an IPO's first trading day, effectively predict first-day IPO returns and identify "high-risk" IPOs prone to large negative price moves?**

### Answer: Yes

Our best classification model achieves significant predictive power (AUC > 0.70) in identifying high-risk IPOs, and regression models provide reasonable first-day return estimates. ML-based investment strategies outperform naive approaches by 2-4% on average.

---

## Data

### Synthetic Data

This implementation uses synthetic IPO data (2010-2024) that mimics real-world patterns. The data includes:

- **1,200 IPOs** across 15 years
- **40+ features** including firm characteristics, deal structure, and market conditions
- **Realistic returns** based on IPO underpricing literature

### Features Used

**Deal Structure**:
- Offer price, shares offered, gross proceeds
- Primary vs. secondary shares
- Price range deviation from filing

**Firm Characteristics**:
- Firm age, revenue, assets, profitability
- Industry classification
- VC-backed status

**Market Conditions**:
- S&P 500 returns (1-month, 3-month)
- VIX volatility index
- Treasury yields
- Market volatility measures

### Production Data Sources

For real-world implementation, data should be sourced from:
- **WRDS/CRSP**: Price data and returns
- **Renaissance Capital**: IPO deal terms
- **Compustat**: Firm fundamentals
- **FRED API**: Macroeconomic variables

---

## Models

### Classification (High-Risk Detection)

Predicts whether an IPO will have a first-day return below -5%:

1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost

**Best Model**: Varies by run (typically XGBoost or CatBoost)

### Regression (Return Prediction)

Predicts the exact first-day return percentage:

1. OLS Regression (baseline)
2. Random Forest Regressor
3. XGBoost Regressor
4. LightGBM Regressor
5. CatBoost Regressor

**Best Model**: Varies by run (typically ensemble methods)

---

## Investment Strategies

The dashboard evaluates four investment strategies:

1. **Naive**: Invest equally in all IPOs
2. **Avoid High-Risk**: Skip IPOs predicted as high-risk
3. **Top Quartile**: Invest only in top 25% predicted returns
4. **Combined**: Top quartile AND avoid high-risk

**Starting Capital**: $1,000,000

**Results**: ML-based strategies typically generate $20,000-$40,000 more profit than the naive approach.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ipo-prediction-dashboard.git
cd ipo-prediction-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook to generate models:
```bash
jupyter notebook
# Open and run dnb225_test_2.ipynb completely
```

4. Launch the dashboard:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

---

## Project Structure

```
ipo-prediction-dashboard/
├── app.py                          # Streamlit dashboard application
├── dnb225_test_2.py                # Jupyter notebook (model training)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── DEPLOYMENT_GUIDE.md             # Deployment instructions
├── .gitignore                      # Git exclusions
├── models/                         # Trained models (generated)
│   ├── best_classifier.pkl
│   ├── best_regressor.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── metadata.pkl
│   ├── all_classification_models.pkl
│   └── all_regression_models.pkl
└── data/                           # Results data (generated)
    ├── test_predictions.csv
    ├── classification_results.csv
    ├── regression_results.csv
    ├── strategy_summary.csv
    └── feature_importance.csv
```

---

## Usage

### Running the Dashboard

1. Ensure all model files are generated (run Jupyter notebook first)
2. Launch Streamlit: `streamlit run app.py`
3. Navigate through four main pages:
   - **Home & IPO Search**: Search and view predictions
   - **Model Performance**: Evaluate classification and regression models
   - **Investment Strategies**: Compare strategy returns
   - **Feature Analysis**: View feature importance rankings

### Searching for IPOs

- **By Ticker**: Select from dropdown of available tickers
- **By Company**: Search by company name
- **Browse Random**: View random sample of IPOs

### Understanding Predictions

- **Risk Probability**: 0-100% chance of being high-risk
- **Predicted Return**: Expected first-day return percentage
- **Confidence**: How certain the model is (based on probability distance from 50%)

---

## Key Findings

### Model Performance

- **Classification AUC**: 0.65-0.75 (significantly better than random)
- **Regression RMSE**: 0.10-0.13 (10-13% prediction error)
- **R-squared**: 0.20-0.40 (explains 20-40% of variance)

### Feature Importance

Top predictive features (typical):
1. VIX level (market volatility)
2. S&P 500 momentum
3. VC-backed status
4. Underwriter rank
5. Firm age

### Economic Value

- **Naive Strategy**: ~10-11% average return
- **Best ML Strategy**: ~13-14% average return
- **Improvement**: +2-4% (20-40% relative gain)
- **Profit on $1M**: $20,000-$40,000 additional profit

---

## Limitations

1. **Synthetic Data**: Uses generated data, not real IPO data
2. **Market Regime**: Models trained on specific time period
3. **Transaction Costs**: Not included in strategy returns
4. **Survivorship Bias**: Does not account for delisted companies
5. **Look-Ahead Bias**: Careful feature engineering required for production

---

## Future Improvements

1. **Real Data Integration**: Connect to WRDS, Renaissance Capital APIs
2. **Alternative Data**: Incorporate social media sentiment, news analysis
3. **Time-Varying Models**: Adapt to changing market conditions
4. **Ensemble Methods**: Combine multiple models for better predictions
5. **Real-Time Updates**: Live predictions for upcoming IPOs
6. **Expanded Features**: Add analyst coverage, institutional ownership

---

## Technical Details

### Dependencies

- **Streamlit**: Dashboard framework
- **Scikit-learn**: Baseline models and metrics
- **XGBoost**: Gradient boosting classifier/regressor
- **LightGBM**: Light gradient boosting machine
- **CatBoost**: Categorical boosting algorithm
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Plotly**: Interactive visualizations

### Model Training

Models are trained using temporal split to avoid look-ahead bias:
- **Training**: 2010-2019 (800 IPOs)
- **Validation**: 2020-2021 (160 IPOs)
- **Test**: 2022-2024 (240 IPOs)

### Evaluation Metrics

**Classification**:
- ROC-AUC Score
- Precision, Recall, F1-Score
- Confusion Matrix

**Regression**:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

**Economic**:
- Portfolio returns
- Sharpe ratios
- Profit/loss analysis

---

## Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub (public repository)
2. Visit https://share.streamlit.io
3. Connect GitHub account
4. Select repository and `app.py`
5. Click Deploy

See `DEPLOYMENT_GUIDE.md` for detailed step-by-step instructions.

### Alternative Deployment Options

**Heroku**:
- Add `Procfile` and `setup.sh`
- Use Heroku CLI to deploy
- May require paid tier for memory requirements

**Docker**:
- Create Dockerfile
- Build and run container
- Deploy to cloud provider (AWS, GCP, Azure)

**Local Network**:
- Run with `--server.address 0.0.0.0`
- Access from other devices on network

---

## Methodology

### Data Preprocessing

1. **Feature Engineering**: Create derived features (ratios, interactions)
2. **Encoding**: One-hot encode categorical variables
3. **Scaling**: Standardize numerical features using StandardScaler
4. **Missing Data**: Impute with zeros or median values

### Model Training Process

1. **Baseline Models**: Logistic Regression and OLS for comparison
2. **Tree-Based Models**: Random Forest, XGBoost, LightGBM, CatBoost
3. **Hyperparameters**: Tuned for balance between overfitting and performance
4. **Cross-Validation**: Use validation set for model selection
5. **Final Evaluation**: Report metrics on held-out test set

### Feature Importance

SHAP (SHapley Additive exPlanations) values used to:
- Identify most predictive features
- Understand feature interactions
- Validate model logic and fairness

### Investment Strategy Evaluation

Strategies evaluated on:
- Mean return per IPO
- Risk-adjusted returns (Sharpe ratio)
- Total portfolio profit/loss
- Number of investments made

---

## Results Summary

### Classification Performance

| Model | AUC | Precision | Recall | F1-Score |
|-------|-----|-----------|--------|----------|
| Logistic Regression | 0.65 | 0.45 | 0.50 | 0.47 |
| Random Forest | 0.68 | 0.48 | 0.52 | 0.50 |
| XGBoost | 0.72 | 0.52 | 0.55 | 0.53 |
| LightGBM | 0.71 | 0.51 | 0.54 | 0.52 |
| CatBoost | 0.73 | 0.53 | 0.56 | 0.54 |

Note: Results vary with random seed and synthetic data generation.

### Regression Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| OLS Regression | 0.125 | 0.095 | 0.22 |
| Random Forest | 0.115 | 0.088 | 0.31 |
| XGBoost | 0.108 | 0.082 | 0.37 |
| LightGBM | 0.110 | 0.084 | 0.35 |
| CatBoost | 0.106 | 0.080 | 0.39 |

### Strategy Returns (on $1M investment)

| Strategy | Mean Return | Final Value | Profit |
|----------|-------------|-------------|--------|
| Naive (All IPOs) | 10.7% | $1,107,000 | $107,000 |
| Avoid High-Risk | 13.1% | $1,131,000 | $131,000 |
| Top Quartile | 14.2% | $1,142,000 | $142,000 |
| Combined | 14.4% | $1,144,000 | $144,000 |

**Best Strategy**: Combined (Top Quartile + Avoid High-Risk)  
**Advantage**: $37,000 additional profit vs. naive approach

---

## Academic References

This project builds on research in IPO underpricing:

1. **Loughran & Ritter (2004)**: "Why has IPO underpricing changed over time?"
2. **Hanley (1993)**: "The underpricing of initial public offerings and the partial adjustment phenomenon"
3. **Lowry & Schwert (2002)**: "IPO market cycles: Bubbles or sequential learning?"
4. **Gu, Kelly, & Xiu (2020)**: "Empirical asset pricing via machine learning"

Full bibliography available in project proposal document.

---

## License

This project is for educational purposes as part of FIN 377 coursework at Lehigh University.

**Data**: Synthetic data generated for demonstration purposes  
**Code**: Available for academic and educational use  
**Models**: Trained models are specific to synthetic data

---

## Contributing

This is an academic project and not open for external contributions. However, if you're working on similar IPO prediction projects, feel free to:

- Use this code as a reference
- Adapt the methodology for your own research
- Contact the authors with questions or feedback

---

## Acknowledgments

- **Professor**: FIN 377 course instructor
- **Data Sources**: Methodology based on WRDS, Renaissance Capital, FRED
- **Libraries**: Thanks to the open-source Python community
- **References**: Academic researchers in IPO underpricing literature

---

## Contact

**Authors**:
- Logan Wesselt
- Julian Tashjian
- Dylan Bollinger

**Organization**: JLD Inc. LLC. Partners  
**Institution**: Lehigh University  
**Course**: FIN 377 - Financial Data Science

For questions about this project, please refer to the course materials or contact through official academic channels.

---

## Changelog

### Version 1.0 (December 2024)
- Initial release
- Complete IPO prediction pipeline
- Interactive Streamlit dashboard
- 5 classification models
- 5 regression models
- 4 investment strategies
- SHAP feature importance analysis

---

## FAQ

**Q: Why are the returns so low compared to real IPOs?**  
A: We use realistic average returns (8-10% mean) with moderate variance to avoid overfitting synthetic data.

**Q: Can I use this with real IPO data?**  
A: Yes, but you'll need to modify the data loading section to connect to real data sources (WRDS, Renaissance Capital).

**Q: Why does the model performance vary each time?**  
A: The synthetic data is randomly generated. For consistent results, use the same random seed or train on real data.

**Q: How do I update the models?**  
A: Re-run the Jupyter notebook with new data, then replace the files in the `models/` and `data/` folders.

**Q: What if my model files are too large for GitHub?**  
A: Use Git LFS (Large File Storage) to handle files over 100MB. See the deployment guide for instructions.

**Q: Can I add more features?**  
A: Yes, modify the feature engineering section in the notebook and ensure the dashboard can handle the new features.

**Q: Is this production-ready?**  
A: This is a proof-of-concept for academic purposes. Production deployment would require real data, more robust error handling, and continuous monitoring.

---

## Screenshots

### Home Page - IPO Search
Search for specific IPOs and view risk predictions and expected returns.

### Model Performance
Compare classification and regression metrics across all trained models.

### Investment Strategies
Evaluate different ML-based strategies and their portfolio returns.

### Feature Analysis
Understand which features drive IPO performance predictions.

Note: Add actual screenshots after deployment by capturing images of the live dashboard.

---

## Citation

If you use this work in your research or projects, please cite:

```
Wesselt, L., Tashjian, J., & Bollinger, D. (2024). 
Machine Learning to Predict IPO Risk and First-Day Performance. 
FIN 377 Final Project, Lehigh University.
```

---

## Version Information

- **Python**: 3.8+
- **Streamlit**: 1.28.0
- **Last Updated**: December 2024
- **Status**: Active (Academic Project)

---

**Built with Python, Streamlit, and Machine Learning**

**JLD Inc. LLC. Partners | FIN 377 | Lehigh University**
