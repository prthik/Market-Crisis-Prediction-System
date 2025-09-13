# Market Crisis Prediction System 🚨

A sophisticated machine learning system for predicting financial market crises with 60+ years of historical data analysis. The system uses ensemble models combining XGBoost, LSTM, and Graph Neural Networks to predict market crashes, financial crises, and systemic shocks.

## Features

- **Real-time Crisis Prediction**: Live monitoring and prediction of market crashes
- **Historical Analysis**: Analyzes 60+ years of market data (1962-2024)
- **Multi-Model Ensemble**: Combines XGBoost, LSTM, and GNN models
- **Web Dashboard**: Interactive Streamlit application for visualization
- **Risk Assessment**: Comprehensive risk factor analysis with actionable recommendations
- **Backtesting**: Historical performance validation across major market events

## System Architecture

```
crisis_prediction_system/
├── app.py                    # Streamlit web application
├── crisis_predictor.py       # Main prediction system
├── models.py                 # ML models (XGBoost, LSTM, GNN)
├── feature_engineering.py    # Feature creation pipeline
├── data_collector.py         # Market data collection
├── config.py                 # System configuration
└── models/                   # Trained model storage
    ├── xgboost_model.pkl
    ├── lstm_model.pt
    └── scalers.pkl
```

## Installation

### Prerequisites
- Python 3.9 or higher
- macOS, Linux, or Windows
- 16GB+ RAM recommended
- 10GB+ free disk space for data

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Prathik_Saduneni_code
```

### Step 2: Set up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Install core dependencies
pip install -r crisis_prediction_system/requirements.txt

# Optional: Install PyTorch for LSTM/GNN models
# For Apple Silicon (M1/M2/M3/M4):
pip install torch torchvision

# For NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Download Historical Data
```bash
# Download the full historical dataset from Kaggle
python download_full_kaggle_dataset.py

# Process the data (this may take 5-10 minutes)
python process_kaggle_full_dataset.py
```

## Quick Start

### 1. Run the Web Application
```bash
cd crisis_prediction_system
streamlit run app.py
```
The application will open in your browser at http://localhost:8501

### 2. Train Models (Optional - pre-trained models included)
```bash
# Train XGBoost model only (fast, no GPU needed)
python train_xgboost_m4.py

# Train full ensemble (requires PyTorch)
python train_lstm_improved_m4.py
```

### 3. Make Command-Line Predictions
```bash
python crisis_predictor.py
```

## Usage Guide

### Web Dashboard Features

1. **Live Predictions Tab**
   - Real-time crisis probability (30-day and 6-month)
   - Risk level indicator (LOW/MEDIUM/HIGH/CRITICAL)
   - Key risk factors analysis
   - Actionable recommendations

2. **Market Analysis Tab**
   - Current market indicators (VIX, yield curve, etc.)
   - Sector performance heatmap
   - Crisis type prediction

3. **Historical Events Tab**
   - Database of major market crises (1962-2024)
   - Event timeline visualization
   - Filter by crisis type

4. **Backtesting Tab**
   - Test model performance on historical data
   - Analyze prediction accuracy by event type

### API Usage

```python
from crisis_prediction_system.crisis_predictor import CrisisPredictionSystem

# Initialize system
predictor = CrisisPredictionSystem()

# Load pre-trained models
predictor.load_models()

# Make live prediction
prediction = predictor.predict_live()

# Access results
print(f"30-Day Crisis Probability: {prediction['predictions']['30_day_probability']:.1%}")
print(f"Risk Level: {prediction['predictions']['risk_level']}")
```

## Model Training

### Dataset Preparation
The system uses the Kaggle "9000 Tickers Stock Market Data" dataset with 34+ million records spanning 1962-2024.

```bash
# 1. Download dataset
python download_full_kaggle_dataset.py

# 2. Process and engineer features
python process_kaggle_full_dataset.py

# 3. Verify data quality
cd crisis_prediction_system
python diagnose_event_labeling.py
```

### Training Models

#### XGBoost Model (Recommended for most users)
```bash
cd crisis_prediction_system
python train_xgboost_m4.py
```
- Training time: 3-5 minutes
- No GPU required
- Achieves 99.7% validation AUC

#### LSTM Model (Requires PyTorch)
```bash
python train_lstm_improved_m4.py
```
- Training time: 30-60 minutes
- GPU recommended (Apple Silicon or NVIDIA)
- Sequential pattern recognition

#### Full Ensemble
```bash
python train_full_ensemble_m4.py
```
- Combines all models
- Best overall performance

## Configuration

Edit `crisis_prediction_system/config.py` to customize:

```python
# Risk thresholds
RISK_THRESHOLDS = {
    'LOW': 0.2,
    'MEDIUM': 0.5,
    'HIGH': 0.7,
    'CRITICAL': 0.85
}

# Model parameters
MODEL_CONFIG = {
    'XGBOOST': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01
    },
    'LSTM': {
        'hidden_size': 128,
        'num_layers': 3,
        'sequence_length': 20
    }
}
```

## Historical Events Database

The system recognizes and learns from 18 major market events:

- **1973-74 Bear Market**: Oil Crisis impact
- **1987 Black Monday**: Single-day crash
- **2000 Dot-com Bubble**: Tech sector collapse
- **2008 Financial Crisis**: Global banking crisis
- **2020 COVID Crash**: Pandemic-driven volatility
- **2023 Banking Crisis**: Regional bank failures
- And 12 more events...

## Performance Metrics

| Model | Validation AUC | Training Time | Hardware |
|-------|---------------|---------------|----------|
| XGBoost | 99.72% | 3 min | CPU |
| LSTM | 98.5% | 45 min | GPU |
| Ensemble | 99.8% | 60 min | GPU |

## Troubleshooting

### PyTorch Import Error
If you see "Tried to instantiate class '__path__._path'":
- The system will automatically fallback to XGBoost-only mode
- To fix: Reinstall PyTorch for your platform
- The app works fine without PyTorch using XGBoost

### Memory Issues
- Reduce batch size in config.py
- Use sampling for LSTM training
- Process data in chunks

### API Key Configuration
Add your API keys to `crisis_prediction_system/config.py`:
```python
API_KEYS = {
    'ALPHA_VANTAGE': 'your_key_here',
    'NEWS_API': 'your_key_here',
    # Optional - system works with mock data if not provided
}
```

## Project Structure

```
Prathik_Saduneni_code/
├── README.md                          # This file
├── crisis_prediction_system/          # Main application
│   ├── app.py                        # Streamlit dashboard
│   ├── crisis_predictor.py           # Core prediction logic
│   ├── models.py                     # ML model definitions
│   ├── feature_engineering.py        # Feature creation
│   ├── data_collector.py             # Data gathering
│   ├── config.py                     # Configuration
│   ├── requirements.txt              # Dependencies
│   └── models/                       # Saved models
├── download_full_kaggle_dataset.py   # Data download
├── process_kaggle_full_dataset.py    # Data processing
└── full_historical_stock_data.parquet # Processed data
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - See LICENSE file for details

## Authors

- Prathik Saduneni - Initial work
- Contributors - Model improvements and bug fixes

## Acknowledgments

- Kaggle for the comprehensive stock market dataset
- Yahoo Finance for real-time market data
- The open-source ML community for libraries and tools

## Contact

For questions or support, please open an issue on GitHub or email me at prathik.saduneni@gmail.com. Happy Training!

---

**Disclaimer**: This system is for educational and research purposes only. Do not use for actual trading decisions without proper risk assessment and professional advice.
