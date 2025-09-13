"""
XGBoost Training Script for M4 Max - Full Dataset
Trains XGBoost model on entire historical dataset with proper parameters
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import time
import psutil
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SYSTEM_CONFIG
from data_collector import HistoricalEventLabeler
from feature_engineering import CrisisFeatureEngineer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb

def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.Process().memory_info().rss / 1024 / 1024 / 1024

def prepare_full_dataset():
    """Prepare the full dataset for XGBoost training"""
    print("=" * 80)
    print("PREPARING FULL DATASET FOR XGBOOST")
    print("=" * 80)
    
    # Load data
    data_path = SYSTEM_CONFIG['HISTORICAL_DATA_PATH']
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records")
    print(f"Memory usage: {get_memory_usage():.1f} GB")
    
    # Label historical events
    event_labeler = HistoricalEventLabeler(data_path)
    df = event_labeler.label_data(df)
    print("✓ Historical events labeled")
    
    # Engineer features
    print("\nEngineering features...")
    feature_engineer = CrisisFeatureEngineer()
    df = feature_engineer.engineer_all_features(df)
    print(f"✓ Created {len(feature_engineer.feature_names)} features")
    
    # Create target
    print("\nCreating target variable...")
    df['crisis_next_30_days'] = df.groupby('Ticker')['is_crisis'].rolling(
        window=30, min_periods=1
    ).max().shift(-30).reset_index(0, drop=True).fillna(0).astype(int)
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_engineer.feature_names if col in numeric_cols]
    
    # Remove label columns
    exclude_cols = ['is_crisis', 'crisis_type', 'crisis_severity', 'days_to_crisis', 
                   'days_in_crisis', 'crisis_next_30_days', 'crisis_next_6_months',
                   'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    # Clean data
    print("\nCleaning data...")
    df_clean = df[feature_cols + ['crisis_next_30_days']].copy()
    
    # Handle infinities
    for col in feature_cols:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    print(f"Removed {initial_rows - len(df_clean):,} rows with NaN values")
    
    print(f"\nFinal dataset:")
    print(f"  Samples: {len(df_clean):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Crisis ratio: {df_clean['crisis_next_30_days'].mean():.2%}")
    print(f"  Memory usage: {get_memory_usage():.1f} GB")
    
    # Force garbage collection
    del df
    gc.collect()
    
    return df_clean, feature_cols

def train_xgboost_full(df_clean, feature_cols):
    """Train XGBoost on full dataset"""
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)
    
    # Prepare data
    X = df_clean[feature_cols].values
    y = df_clean['crisis_next_30_days'].values
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Memory usage after scaling: {get_memory_usage():.1f} GB")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Positive class ratio - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")
    
    # XGBoost parameters optimized for M4 Max
    n_cores = psutil.cpu_count(logical=False)
    print(f"\nUsing {n_cores} CPU cores")
    
    # Create DMatrix for efficient memory usage
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Clear original arrays to save memory
    del X_train, X_val
    gc.collect()
    
    # Parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': (1 - y_train.mean()) / y_train.mean(),  # Handle class imbalance
        'tree_method': 'hist',  # Efficient for M-series
        'nthread': n_cores,
        'seed': 42
    }
    
    # Training
    print("\nTraining XGBoost...")
    print("Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    start_time = time.time()
    
    # Callbacks for progress
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=25
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Evaluate
    print("\nEvaluating model...")
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    
    train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
    val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
    
    print(f"\nPerformance Metrics:")
    print(f"  Training AUC: {train_auc:.4f}")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    
    # Classification report
    print("\nValidation Set Classification Report:")
    print(classification_report(y_val, (val_pred > 0.5).astype(int), 
                              target_names=['No Crisis', 'Crisis']))
    
    # Feature importance
    print("\nTop 20 Most Important Features:")
    importance = model.get_score(importance_type='gain')
    feature_importance = {}
    for k, v in importance.items():
        if k.startswith('f'):
            idx = int(k[1:])
            if idx < len(feature_cols):
                feature_importance[feature_cols[idx]] = v
    
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_features[:20]):
        print(f"  {i+1}. {feat}: {score:.2f}")
    
    # Save model
    print("\nSaving model and metadata...")
    
    # Save XGBoost native format
    model.save_model('models/xgboost_native.json')
    
    # Create sklearn-compatible wrapper for compatibility
    sklearn_params = params.copy()
    sklearn_params.pop('eval_metric', None)  # Remove non-sklearn param
    sklearn_params['n_estimators'] = model.best_iteration
    sklearn_params['n_jobs'] = sklearn_params.pop('nthread')
    
    sklearn_model = xgb.XGBClassifier(**sklearn_params)
    sklearn_model.fit(scaler.transform(df_clean[feature_cols].values), y, verbose=False)
    joblib.dump(sklearn_model, 'models/xgboost_model.pkl')
    
    # Save scaler
    scalers = {'xgboost': scaler}
    joblib.dump(scalers, 'models/scalers.pkl')
    
    # Save feature columns
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)
    
    # Save metadata
    metadata = {
        'model': 'XGBoost',
        'training_date': str(datetime.now()),
        'dataset_size': len(df_clean),
        'n_features': len(feature_cols),
        'training_time_minutes': training_time / 60,
        'performance': {
            'train_auc': float(train_auc),
            'val_auc': float(val_auc),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc)
        },
        'best_iteration': model.best_iteration,
        'feature_importance': dict(sorted_features[:50]),
        'device': 'Apple M4 Max',
        'parameters': params
    }
    
    with open('models/xgboost_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ XGBoost model saved successfully!")
    
    return model, scaler, feature_importance

def main():
    """Main training function"""
    print("=" * 80)
    print("XGBOOST TRAINING ON FULL DATASET - M4 MAX")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print(f"Initial memory usage: {get_memory_usage():.1f} GB")
    
    try:
        # Prepare data
        df_clean, feature_cols = prepare_full_dataset()
        
        # Train model
        model, scaler, feature_importance = train_xgboost_full(df_clean, feature_cols)
        
        print("\n" + "=" * 80)
        print("✅ XGBOOST TRAINING COMPLETE!")
        print("=" * 80)
        print("Model saved to: models/xgboost_model.pkl")
        print("Native format: models/xgboost_native.json")
        print("Metadata: models/xgboost_metadata.json")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now()}")
    print(f"Final memory usage: {get_memory_usage():.1f} GB")

if __name__ == "__main__":
    main()
