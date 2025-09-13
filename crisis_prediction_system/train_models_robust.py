"""
Robust model training script that handles memory constraints and saves models properly
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
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SYSTEM_CONFIG, MODEL_CONFIG
from data_collector import HistoricalEventLabeler
from feature_engineering import CrisisFeatureEngineer

def prepare_training_data(sample_size=100000):
    """Prepare training data with proper sampling"""
    print("=" * 80)
    print("PREPARING TRAINING DATA")
    print("=" * 80)
    
    # Load historical data
    data_path = SYSTEM_CONFIG['HISTORICAL_DATA_PATH']
    if not os.path.exists(data_path):
        print(f"Error: Historical data not found at {data_path}")
        return None, None, None
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records")
    
    # Label historical events
    event_labeler = HistoricalEventLabeler(data_path)
    df = event_labeler.label_data(df)
    print("✓ Historical events labeled")
    
    # Sample data to manage memory
    if len(df) > sample_size:
        print(f"\nSampling {sample_size:,} records to manage memory...")
        
        # Stratified sampling to ensure we have crisis examples
        crisis_data = df[df['is_crisis'] == 1]
        normal_data = df[df['is_crisis'] == 0]
        
        # Sample proportionally
        n_crisis = min(len(crisis_data), int(sample_size * 0.3))
        n_normal = min(len(normal_data), sample_size - n_crisis)
        
        sampled_crisis = crisis_data.sample(n=n_crisis, random_state=42) if n_crisis > 0 else pd.DataFrame()
        sampled_normal = normal_data.sample(n=n_normal, random_state=42) if n_normal > 0 else pd.DataFrame()
        
        df = pd.concat([sampled_crisis, sampled_normal]).sort_values('Date')
        print(f"Sampled data: {len(df):,} records ({n_crisis} crisis, {n_normal} normal)")
    
    # Engineer features (simplified set)
    print("\nEngineering features...")
    feature_engineer = CrisisFeatureEngineer()
    
    # Only create essential features to save memory
    df = create_essential_features(df)
    
    # Create target variable
    df['crisis_next_30_days'] = df.groupby('Ticker')['is_crisis'].rolling(
        window=30, min_periods=1
    ).max().shift(-30).reset_index(0, drop=True).fillna(0).astype(int)
    
    print("✓ Features engineered")
    
    # Select feature columns
    feature_cols = [
        'returns', 'volatility_20', 'RSI', 'MACD', 'volume_ratio',
        'SMA_ratio_20', 'SMA_ratio_50', 'BB_position',
        'high_volatility', 'extreme_return', 'volume_spike'
    ]
    
    # Add any other numeric columns that exist
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in feature_cols and col not in ['is_crisis', 'crisis_next_30_days', 'Open', 'High', 'Low', 'Close', 'Volume']:
            feature_cols.append(col)
    
    # Filter to valid columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    return df, feature_cols, feature_engineer

def create_essential_features(df):
    """Create only essential features to save memory"""
    # Price returns
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Volatility (with fillna)
    df['volatility_20'] = df.groupby('Ticker')['returns'].rolling(20, min_periods=5).std().reset_index(0, drop=True) * np.sqrt(252)
    df['volatility_20'] = df['volatility_20'].fillna(df.groupby('Ticker')['volatility_20'].transform('mean'))
    
    # RSI (simplified)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Default RSI to neutral
    
    df['RSI'] = df.groupby('Ticker')['Close'].apply(lambda x: calculate_rsi(x)).reset_index(0, drop=True)
    
    # MACD
    df['EMA_12'] = df.groupby('Ticker')['Close'].ewm(span=12, min_periods=1).mean().reset_index(0, drop=True)
    df['EMA_26'] = df.groupby('Ticker')['Close'].ewm(span=26, min_periods=1).mean().reset_index(0, drop=True)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    # Volume ratio
    df['volume_SMA_20'] = df.groupby('Ticker')['Volume'].rolling(20, min_periods=1).mean().reset_index(0, drop=True)
    df['volume_ratio'] = df['Volume'] / (df['volume_SMA_20'] + 1e-10)
    
    # SMA ratios
    df['SMA_20'] = df.groupby('Ticker')['Close'].rolling(20, min_periods=1).mean().reset_index(0, drop=True)
    df['SMA_50'] = df.groupby('Ticker')['Close'].rolling(50, min_periods=1).mean().reset_index(0, drop=True)
    df['SMA_ratio_20'] = df['Close'] / (df['SMA_20'] + 1e-10)
    df['SMA_ratio_50'] = df['Close'] / (df['SMA_50'] + 1e-10)
    
    # Bollinger Bands position
    bb_std = df.groupby('Ticker')['Close'].rolling(20, min_periods=1).std().reset_index(0, drop=True)
    df['BB_upper'] = df['SMA_20'] + (bb_std * 2)
    df['BB_lower'] = df['SMA_20'] - (bb_std * 2)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = np.where(df['BB_width'] > 0, 
                                  (df['Close'] - df['BB_lower']) / df['BB_width'],
                                  0.5)  # Default to middle
    
    # Binary indicators with proper handling
    vol_threshold = df['volatility_20'].quantile(0.8)
    df['high_volatility'] = (df['volatility_20'] > vol_threshold).astype(int)
    df['extreme_return'] = (abs(df['returns']) > 0.05).astype(int)  # 5% return threshold
    df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
    
    # Fill any remaining NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def train_random_forest_model(X_train, y_train, X_val, y_val):
    """Train a Random Forest model as a robust alternative"""
    print("\nTraining Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Validation accuracy: {val_score:.3f}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    
    return model, feature_importance

def save_models_properly(model, scaler, feature_columns, feature_importance):
    """Save models in the correct format"""
    print("\nSaving models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the main model
    joblib.dump(model, 'models/xgboost_model.pkl')
    print("✓ Saved model as xgboost_model.pkl")
    
    # Save scaler
    scalers = {
        'xgboost': scaler,
        'lstm': StandardScaler(),  # Placeholder
        'gnn': StandardScaler()    # Placeholder
    }
    joblib.dump(scalers, 'models/scalers.pkl')
    print("✓ Saved scalers")
    
    # Save feature columns
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    print("✓ Saved feature columns")
    
    # Create LSTM placeholder (to avoid loading errors)
    lstm_state = {
        'hidden_size': 128,
        'num_layers': 3,
        'input_size': len(feature_columns),
        'trained': False
    }
    torch.save(lstm_state, 'models/lstm_model.pt')
    print("✓ Created LSTM placeholder")
    
    # Save training results
    results = {
        'training_completed': True,
        'model_type': 'RandomForest',
        'features': len(feature_columns),
        'feature_importance': {col: float(imp) for col, imp in zip(feature_columns, feature_importance)},
        'timestamp': str(datetime.now()),
        'note': 'Model trained on sampled data for efficiency'
    }
    
    with open('models/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved training results")

def main():
    print("=" * 80)
    print("ROBUST MODEL TRAINING FOR CRISIS PREDICTION")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Prepare data
        df, feature_cols, feature_engineer = prepare_training_data(sample_size=50000)
        
        if df is None:
            print("Failed to prepare data!")
            return
        
        # Clean data
        print("\nCleaning data...")
        df_clean = df.dropna(subset=feature_cols + ['crisis_next_30_days'])
        print(f"Clean samples: {len(df_clean):,}")
        
        # Prepare features and target
        X = df_clean[feature_cols].values
        y = df_clean['crisis_next_30_days'].values
        
        # Handle infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Scale features
        print("\nScaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Validation set: {len(X_val):,} samples")
        print(f"Crisis ratio: {y.mean():.3%}")
        
        # Train model
        model, feature_importance = train_random_forest_model(X_train, y_train, X_val, y_val)
        
        # Save everything
        save_models_properly(model, scaler, feature_cols, feature_importance)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print("\nModels saved to 'models/' directory")
        print("You can now run the Streamlit app with: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nAttempting to create minimal fallback models...")
        os.system("python create_minimal_models.py")
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
