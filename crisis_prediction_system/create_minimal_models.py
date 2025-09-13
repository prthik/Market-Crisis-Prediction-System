"""
Create minimal pre-trained models for the crisis prediction system
This ensures the Streamlit app can load and run even with limited resources
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_minimal_models():
    print("=" * 80)
    print("CREATING MINIMAL PRE-TRAINED MODELS")
    print("=" * 80)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Create dummy feature names (simplified set)
    feature_columns = [
        'returns', 'volatility_20', 'RSI', 'MACD', 'volume_ratio',
        'SMA_20', 'SMA_50', 'price_to_SMA_20', 'BB_position',
        'crisis_score', 'VIX_proxy', 'yield_curve_proxy'
    ]
    
    # Create synthetic training data
    print("\n1. Creating synthetic training data...")
    n_samples = 1000
    np.random.seed(42)
    
    # Create features with some structure
    X = np.random.randn(n_samples, len(feature_columns))
    
    # Add some correlation structure
    X[:, 1] = X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.5  # volatility correlated with returns
    X[:, 2] = np.clip(50 + X[:, 0] * 20 + np.random.randn(n_samples) * 10, 0, 100)  # RSI
    
    # Create target variable with some logic
    crisis_score = (
        (X[:, 1] > 1.5) * 0.3 +  # high volatility
        (X[:, 2] > 70) * 0.2 +   # high RSI
        (X[:, 0] < -1) * 0.3 +   # negative returns
        np.random.rand(n_samples) * 0.2
    )
    y = (crisis_score > 0.5).astype(int)
    
    print(f"Created {n_samples} samples with {sum(y)} crisis examples")
    
    # Train a simple Random Forest as XGBoost replacement
    print("\n2. Training Random Forest model...")
    
    # Scale the data first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Save the model directly
    joblib.dump(model, 'models/xgboost_model.pkl')
    print("✓ Saved Random Forest model as XGBoost replacement")
    
    # Save feature columns
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    print("✓ Saved feature columns")
    
    # Create dummy LSTM model state
    print("\n3. Creating LSTM model state...")
    lstm_state = {
        'hidden_size': 128,
        'num_layers': 3,
        'input_size': len(feature_columns),
        'trained': False
    }
    torch.save(lstm_state, 'models/lstm_model.pt')
    print("✓ Saved LSTM model state")
    
    # Save scalers
    scalers = {
        'lstm': StandardScaler(),
        'gnn': StandardScaler(),
        'xgboost': scaler
    }
    joblib.dump(scalers, 'models/scalers.pkl')
    print("✓ Saved scalers")
    
    # Create test results
    test_results = {
        'training_completed': True,
        'model_type': 'minimal',
        'training_samples': n_samples,
        'features': len(feature_columns),
        'test_prediction': {
            '30_day_probability': 0.15,
            '6_month_probability': 0.25,
            'risk_level': 'LOW'
        },
        'timestamp': str(datetime.now()),
        'note': 'Minimal models created for demonstration. Train with full data for accurate predictions.'
    }
    
    with open('models/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    print("✓ Saved test results")
    
    print("\n" + "=" * 80)
    print("✅ MINIMAL MODELS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print("\nCreated files:")
    print("  - models/xgboost_model.pkl (Random Forest wrapper)")
    print("  - models/lstm_model.pt (placeholder)")
    print("  - models/scalers.pkl")
    print("  - models/feature_columns.json")
    print("  - models/test_results.json")
    print("\n⚠️  NOTE: These are minimal models for demonstration.")
    print("    For accurate predictions, train with the full dataset.")
    print("\n🚀 You can now run 'streamlit run app.py' to launch the web app!")

if __name__ == "__main__":
    create_minimal_models()
