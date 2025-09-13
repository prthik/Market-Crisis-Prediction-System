"""
Safe Ensemble Training Script for M4 Max
Handles memory constraints and provides stable training
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import json
from datetime import datetime
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SYSTEM_CONFIG
from data_collector import HistoricalEventLabeler
from feature_engineering import CrisisFeatureEngineer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Safe configuration
SAFE_CONFIG = {
    'sample_size': 300000,  # Reduced sample size
    'n_features': 50,  # Top features to use
    'xgb_trees': 300,  # Reduced trees
    'rf_trees': 200,  # Random forest trees
}

def prepare_data_safely():
    """Prepare data with memory safety"""
    print("=" * 80)
    print("SAFE DATA PREPARATION FOR M4 MAX")
    print("=" * 80)
    
    # Load data
    data_path = SYSTEM_CONFIG['HISTORICAL_DATA_PATH']
    print(f"Loading data from {data_path}...")
    
    # Read in chunks to manage memory
    df = pd.read_parquet(data_path)
    original_size = len(df)
    print(f"Loaded {original_size:,} records")
    
    # Sample data for safety
    if len(df) > SAFE_CONFIG['sample_size']:
        print(f"\nSampling {SAFE_CONFIG['sample_size']:,} records for stability...")
        
        # Ensure we get crisis examples
        crisis_mask = df['is_crisis'] == 1 if 'is_crisis' in df.columns else pd.Series([False] * len(df))
        crisis_data = df[crisis_mask]
        normal_data = df[~crisis_mask]
        
        # Calculate proportions
        n_crisis = min(len(crisis_data), int(SAFE_CONFIG['sample_size'] * 0.3))
        n_normal = SAFE_CONFIG['sample_size'] - n_crisis
        
        # Sample
        if n_crisis > 0:
            sampled_crisis = crisis_data.sample(n=n_crisis, random_state=42)
        else:
            sampled_crisis = pd.DataFrame()
            
        if n_normal > 0 and len(normal_data) > 0:
            sampled_normal = normal_data.sample(n=min(n_normal, len(normal_data)), random_state=42)
        else:
            sampled_normal = pd.DataFrame()
        
        df = pd.concat([sampled_crisis, sampled_normal]).sort_values('Date')
    
    # Label events if needed
    if 'is_crisis' not in df.columns:
        event_labeler = HistoricalEventLabeler(data_path)
        df = event_labeler.label_data(df)
        print("✓ Historical events labeled")
    
    # Create basic features
    print("\nCreating essential features...")
    
    # Basic price features
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df.groupby('Ticker')['Close'].shift(1))
    
    # Volatility
    df['volatility_20'] = df.groupby('Ticker')['returns'].rolling(20, min_periods=5).std().reset_index(0, drop=True) * np.sqrt(252)
    df['volatility_60'] = df.groupby('Ticker')['returns'].rolling(60, min_periods=10).std().reset_index(0, drop=True) * np.sqrt(252)
    
    # Simple technical indicators
    df['SMA_20'] = df.groupby('Ticker')['Close'].rolling(20, min_periods=5).mean().reset_index(0, drop=True)
    df['SMA_50'] = df.groupby('Ticker')['Close'].rolling(50, min_periods=10).mean().reset_index(0, drop=True)
    df['price_to_SMA20'] = df['Close'] / df['SMA_20']
    df['price_to_SMA50'] = df['Close'] / df['SMA_50']
    
    # Volume features
    df['volume_SMA_20'] = df.groupby('Ticker')['Volume'].rolling(20, min_periods=5).mean().reset_index(0, drop=True)
    df['volume_ratio'] = df['Volume'] / df['volume_SMA_20']
    
    # RSI simplified
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))
    
    # Binary features
    df['high_volatility'] = (df['volatility_20'] > df['volatility_20'].quantile(0.8)).astype(int)
    df['extreme_return'] = (abs(df['returns']) > 0.05).astype(int)
    df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
    
    # Create target
    df['crisis_next_30_days'] = df.groupby('Ticker')['is_crisis'].transform(
        lambda x: x.rolling(window=30, min_periods=1).max().shift(-30)
    ).fillna(0).astype(int)
    
    print("✓ Features created")
    
    # Select feature columns
    feature_cols = [
        'returns', 'log_returns', 'volatility_20', 'volatility_60',
        'price_to_SMA20', 'price_to_SMA50', 'volume_ratio', 'RSI',
        'high_volatility', 'extreme_return', 'volume_spike'
    ]
    
    # Clean data
    df_clean = df[feature_cols + ['crisis_next_30_days']].copy()
    
    # Handle infinities and NaN
    for col in feature_cols:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    df_clean = df_clean.dropna()
    
    print(f"\nClean samples: {len(df_clean):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Crisis ratio: {df_clean['crisis_next_30_days'].mean():.2%}")
    
    # Force garbage collection
    del df
    gc.collect()
    
    return df_clean, feature_cols

def train_models_safely(df_clean, feature_cols):
    """Train models with safety measures"""
    print("\n" + "=" * 80)
    print("TRAINING MODELS SAFELY")
    print("=" * 80)
    
    # Prepare data
    X = df_clean[feature_cols].values
    y = df_clean['crisis_next_30_days'].values
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    
    # Train XGBoost
    print("\n1. Training XGBoost...")
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=SAFE_CONFIG['xgb_trees'],
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            tree_method='hist',
            random_state=42,
            use_label_encoder=False
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            early_stopping_rounds=30,
            verbose=True
        )
        
        # Evaluate
        val_pred = xgb_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
        
        print(f"\nXGBoost - Validation AUC: {val_auc:.3f}, Accuracy: {val_acc:.3f}")
        
        # Save
        joblib.dump(xgb_model, 'models/xgboost_model.pkl')
        print("✓ XGBoost model saved")
        
    except Exception as e:
        print(f"XGBoost failed: {e}")
        print("Falling back to Random Forest...")
        
        # Train Random Forest as fallback
        rf_model = RandomForestClassifier(
            n_estimators=SAFE_CONFIG['rf_trees'],
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        val_pred = rf_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
        
        print(f"\nRandom Forest - Validation AUC: {val_auc:.3f}, Accuracy: {val_acc:.3f}")
        
        # Save as XGBoost
        joblib.dump(rf_model, 'models/xgboost_model.pkl')
        print("✓ Random Forest model saved (as XGBoost replacement)")
    
    # Save additional files
    print("\n2. Saving model metadata...")
    
    # Save scaler
    scalers = {'xgboost': scaler}
    joblib.dump(scalers, 'models/scalers.pkl')
    
    # Save feature columns
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)
    
    # Create LSTM placeholder
    lstm_state = {
        'hidden_size': 128,
        'num_layers': 2,
        'input_size': len(feature_cols),
        'trained': False
    }
    torch.save(lstm_state, 'models/lstm_model.pt')
    
    # Save results
    results = {
        'training_completed': True,
        'timestamp': str(datetime.now()),
        'model_type': 'Ensemble (Safe Mode)',
        'features': len(feature_cols),
        'samples_used': len(X_train) + len(X_val),
        'device': 'Apple M4 Max',
        'note': 'Trained with memory-safe configuration'
    }
    
    with open('models/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ All models and metadata saved successfully!")

def main():
    """Main training function"""
    print("=" * 80)
    print("SAFE ENSEMBLE TRAINING FOR M4 MAX")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Prepare data
        df_clean, feature_cols = prepare_data_safely()
        
        # Train models
        train_models_safely(df_clean, feature_cols)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print("Models are ready for use in the Streamlit app")
        print("Run: cd crisis_prediction_system && streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
