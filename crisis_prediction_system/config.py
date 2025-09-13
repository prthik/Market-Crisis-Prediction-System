"""
Configuration file for Crisis Prediction System
Contains API keys and system settings
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
API_KEYS = {
    'ALPHA_VANTAGE': 'U9NKG72SGJNV5UVN',
    'POLYGON': 'WAVrpFv6jyvD651RLbwIFNDuioWOqdSU',
    'FRED': 'f0458bbcb81b9976f15b7b07f22110ac',
    'NEWS_API': 'cd139d51ea1143ab85ac2d2e5a226e0b',
    'OPENEXCHANGE': 'a19f828518ea41eea1f8dce4fb91168d'
}

# System Configuration
SYSTEM_CONFIG = {
    'PREDICTION_HORIZONS': {
        'SHORT_TERM': 30,  # 30 days
        'LONG_TERM': 180   # 6 months
    },
    'RISK_THRESHOLDS': {
        'LOW': 0.2,
        'MEDIUM': 0.5,
        'HIGH': 0.7,
        'CRITICAL': 0.85
    },
    'UPDATE_FREQUENCY': 3600,  # Update every hour
    'HISTORICAL_DATA_PATH': '../full_processed_stock_data.parquet',
    'MODEL_PATH': 'models/',
    'DATA_PATH': 'data/',
    'LOG_PATH': 'logs/'
}

# Market Events Categories
EVENT_CATEGORIES = {
    'FINANCIAL_CRISIS': {
        'description': 'Banking/credit system failures',
        'examples': ['2008 GFC', '1998 LTCM', 'S&L Crisis'],
        'severity_weight': 0.9
    },
    'MARKET_CRASH': {
        'description': 'Rapid market declines >20%',
        'examples': ['1987 Black Monday', '2020 COVID Crash', 'Dotcom Bust'],
        'severity_weight': 0.85
    },
    'SECTOR_BUBBLE': {
        'description': 'Sector-specific bubbles bursting',
        'examples': ['2000 Tech Bubble', '2021 SPAC Bubble', 'Crypto Winter'],
        'severity_weight': 0.6
    },
    'GEOPOLITICAL_SHOCK': {
        'description': 'War, terrorism, political upheaval',
        'examples': ['9/11', 'Ukraine War', 'Trade Wars'],
        'severity_weight': 0.7
    },
    'CURRENCY_CRISIS': {
        'description': 'Major currency devaluations',
        'examples': ['1997 Asian Crisis', '1998 Russian Default'],
        'severity_weight': 0.65
    },
    'COMMODITY_SHOCK': {
        'description': 'Oil, gold, commodity price shocks',
        'examples': ['1973 Oil Crisis', '2020 Negative Oil'],
        'severity_weight': 0.5
    },
    'PANDEMIC': {
        'description': 'Global health crises',
        'examples': ['COVID-19', 'SARS', 'H1N1'],
        'severity_weight': 0.8
    },
    'POLICY_SHOCK': {
        'description': 'Major policy changes',
        'examples': ['Volcker Shock', 'Brexit', 'Fed Tightening'],
        'severity_weight': 0.55
    }
}

# Feature Configuration
FEATURE_CONFIG = {
    'TECHNICAL_INDICATORS': [
        'RSI', 'MACD', 'BB_Position', 'ATR', 'Volume_Ratio',
        'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26'
    ],
    'VOLATILITY_MEASURES': [
        'Volatility_20', 'Volatility_60', 'VIX', 'VVIX',
        'Realized_Vol', 'Implied_Vol', 'Vol_of_Vol'
    ],
    'CORRELATION_FEATURES': [
        'Cross_Asset_Correlation', 'Sector_Correlation',
        'Rolling_Correlation', 'Correlation_Breakdown'
    ],
    'MARKET_MICROSTRUCTURE': [
        'Bid_Ask_Spread', 'Volume_Imbalance', 'Tick_Direction',
        'Order_Flow', 'Large_Trades'
    ],
    'ECONOMIC_INDICATORS': [
        'Yield_Curve', 'Term_Spread', 'Credit_Spread',
        'Dollar_Index', 'Inflation_Expectations', 'GDP_Nowcast'
    ],
    'SENTIMENT_INDICATORS': [
        'News_Sentiment', 'Fear_Greed_Index', 'Put_Call_Ratio',
        'Breadth_Indicators', 'Google_Trends'
    ]
}

# Model Configuration
MODEL_CONFIG = {
    'LSTM': {
        'sequence_length': 60,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.001
    },
    'GNN': {
        'num_nodes': 100,  # Top 100 stocks
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.3
    },
    'XGBOOST': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8
    },
    'ENSEMBLE': {
        'weights': {
            'LSTM': 0.4,
            'GNN': 0.3,
            'XGBOOST': 0.3
        }
    }
}
