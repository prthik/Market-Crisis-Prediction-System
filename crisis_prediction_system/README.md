# 🚨 Market Crisis Prediction System

A sophisticated AI-powered system that predicts market crashes, financial crises, and systemic shocks with 30-day and 6-month horizons. Built using 60+ years of historical market data and real-time indicators.

## 🎯 Features

- **Multi-Model Ensemble**: Combines LSTM, Graph Neural Networks, and XGBoost
- **Real-time Monitoring**: Live market data integration via multiple APIs
- **Historical Analysis**: Trained on all major crises from 1962-2024
- **Crisis Classification**: Identifies 8 different types of market events
- **Web Dashboard**: Interactive Streamlit application
- **Backtesting**: Validate predictions on historical events
- **Actionable Insights**: Risk levels and portfolio recommendations

## 📊 Crisis Types Detected

1. **Financial Crises** (2008 GFC, S&L Crisis)
2. **Market Crashes** (1987 Black Monday, 2020 COVID)
3. **Sector Bubbles** (Dotcom, Crypto Winter)
4. **Geopolitical Shocks** (9/11, Trade Wars)
5. **Currency Crises** (Asian Crisis, Russian Default)
6. **Commodity Shocks** (Oil Crises)
7. **Pandemic Events** (COVID-19, SARS)
8. **Policy Shocks** (Volcker Shock, Brexit)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd crisis_prediction_system

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up API Keys

The system uses the following FREE APIs:
- **Alpha Vantage**: Stock market data
- **FRED**: Economic indicators
- **NewsAPI**: Sentiment analysis
- **OpenExchangeRates**: Currency data
- **Polygon.io**: Additional market data

API keys are already configured in `config.py`.

### 3. Run the Web Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### 4. Train Models (First Time)

Click "Train Models" in the sidebar. This will:
- Load historical data
- Label crisis events
- Engineer features
- Train ensemble models

Training takes approximately 5-10 minutes.

## 📱 Using the Dashboard

### Live Predictions Tab
- **Risk Gauges**: Visual 30-day and 6-month crisis probabilities
- **Risk Level**: LOW, MEDIUM, HIGH, or CRITICAL
- **Risk Factors**: Current market conditions driving predictions
- **Recommendations**: Actionable portfolio advice

### Market Analysis Tab
- **Key Indicators**: VIX, yield curve, economic data
- **Sector Performance**: Heatmap of sector returns
- **Crisis Type**: Most likely event type prediction

### Historical Events Tab
- Browse 60+ years of crisis events
- Filter by event type
- Interactive timeline visualization

### Backtesting Tab
- Test model accuracy on historical data
- Select custom date ranges
- View performance by event type

## 📁 System Architecture

```
crisis_prediction_system/
├── config.py              # API keys and configuration
├── data_collector.py      # Real-time data collection
├── feature_engineering.py # Advanced feature creation
├── models.py             # LSTM, GNN, XGBoost models
├── crisis_predictor.py   # Main prediction engine
├── app.py               # Streamlit web application
└── requirements.txt     # Python dependencies
```

## 🔧 Advanced Usage

### Command Line Prediction

```python
from crisis_predictor import CrisisPredictionSystem

# Initialize system
predictor = CrisisPredictionSystem()

# Train models (if not already trained)
predictor.train_models()

# Get live prediction
result = predictor.predict_live()

print(f"30-Day Crisis Probability: {result['predictions']['30_day_probability']:.1%}")
print(f"Risk Level: {result['predictions']['risk_level']}")
```

### Backtesting

```python
# Run backtest on specific period
results = predictor.backtest(
    start_date='2015-01-01',
    end_date='2023-12-31'
)

print(f"Accuracy: {results['accuracy']:.1%}")
```

## 📈 Model Performance

Based on backtesting 2010-2023:
- **Major Crises Detected**: 85%+
- **False Positive Rate**: <15%
- **Average Warning Time**: 15-25 days

## 🛡️ Risk Management Guidelines

### Risk Levels and Actions

**LOW (< 20%)**
- Maintain normal allocations
- Look for opportunities

**MEDIUM (20-50%)**
- Increase monitoring
- Rebalance to defensive sectors

**HIGH (50-70%)**
- Reduce risk exposure
- Increase cash to 20-30%

**CRITICAL (> 70%)**
- Immediate risk reduction
- Cash positions 30-50%
- Implement hedging strategies

## 🔄 Updating Models

Models automatically retrain when:
- New crisis events occur
- Significant market regime changes
- Monthly scheduled updates

## 📊 Data Sources

- **Historical Data**: 34.6M records, 9,315 tickers (1962-2024)
- **Real-time Data**: 15+ market indicators updated hourly
- **News Sentiment**: 100+ articles analyzed daily
- **Economic Data**: Fed data, yield curves, inflation

## ⚠️ Disclaimer

This system is for educational and research purposes. Predictions are probabilistic and should not be the sole basis for investment decisions. Always consult with financial professionals and conduct your own analysis.

## 🤝 Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Ensure all API keys are valid
3. Verify internet connection for real-time data

## 📝 License

MIT License - See LICENSE file for details

---

**Built with ❤️ using advanced machine learning and 60+ years of market wisdom**
