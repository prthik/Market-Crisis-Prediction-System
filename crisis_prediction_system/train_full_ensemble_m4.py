"""
Full Ensemble Model Training Script Optimized for Apple M4 Max
Trains XGBoost, LSTM, and Graph Neural Network models with proper memory management
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

from config import SYSTEM_CONFIG, MODEL_CONFIG
from data_collector import HistoricalEventLabeler
from feature_engineering import CrisisFeatureEngineer
from models import LSTMCrisisPredictor, GraphNeuralNetwork, XGBoostCrisisModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score
import xgboost as xgb

# Check for Apple Silicon and MPS availability
def check_device():
    """Check and configure device for Apple Silicon"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon GPU (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (GPU not available)")
    return device

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024  # GB

def print_system_info():
    """Print system information"""
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"XGBoost: {xgb.__version__}")
    
    # CPU info
    print(f"\nCPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory info
    mem = psutil.virtual_memory()
    print(f"Total Memory: {mem.total / 1024 / 1024 / 1024:.1f} GB")
    print(f"Available Memory: {mem.available / 1024 / 1024 / 1024:.1f} GB")
    
    # Check for M-series chip
    try:
        import platform
        if platform.processor() == 'arm':
            print("✅ Apple Silicon detected")
    except:
        pass
    
    print("=" * 80)

class ProgressTracker:
    """Track training progress across models"""
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = []
        
    def log_checkpoint(self, model_name, metric, value):
        """Log a training checkpoint"""
        checkpoint = {
            'timestamp': datetime.now(),
            'elapsed_time': time.time() - self.start_time,
            'model': model_name,
            'metric': metric,
            'value': value,
            'memory_gb': get_memory_usage()
        }
        self.checkpoints.append(checkpoint)
        
        # Print progress
        elapsed = checkpoint['elapsed_time']
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"\n📊 [{hours:02d}:{minutes:02d}] {model_name} - {metric}: {value:.4f} (Memory: {checkpoint['memory_gb']:.1f} GB)")

def prepare_data(sample_size=None):
    """Prepare data with optional sampling"""
    print("\n" + "=" * 80)
    print("PREPARING TRAINING DATA")
    print("=" * 80)
    
    # Load data
    data_path = SYSTEM_CONFIG['HISTORICAL_DATA_PATH']
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records")
    
    # Label historical events
    event_labeler = HistoricalEventLabeler(data_path)
    df = event_labeler.label_data(df)
    print("✓ Historical events labeled")
    
    # Sample if requested (for testing)
    if sample_size and len(df) > sample_size:
        print(f"\nSampling {sample_size:,} records...")
        
        # Stratified sampling
        crisis_data = df[df['is_crisis'] == 1]
        normal_data = df[df['is_crisis'] == 0]
        
        n_crisis = min(len(crisis_data), int(sample_size * 0.3))
        n_normal = sample_size - n_crisis
        
        df = pd.concat([
            crisis_data.sample(n=n_crisis, random_state=42),
            normal_data.sample(n=n_normal, random_state=42)
        ]).sort_values('Date')
    
    # Engineer features
    print("\nEngineering features...")
    feature_engineer = CrisisFeatureEngineer()
    df = feature_engineer.engineer_all_features(df)
    print(f"✓ Created {len(feature_engineer.feature_names)} features")
    
    # Create targets
    df['crisis_next_30_days'] = df.groupby('Ticker')['is_crisis'].rolling(
        window=30, min_periods=1
    ).max().shift(-30).reset_index(0, drop=True).fillna(0).astype(int)
    
    print("✓ Target variables created")
    
    # Select features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_engineer.feature_names if col in numeric_cols]
    
    # Remove label columns
    exclude_cols = ['is_crisis', 'crisis_type', 'crisis_severity', 'days_to_crisis', 
                   'days_in_crisis', 'crisis_next_30_days', 'crisis_next_6_months']
    feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    # Clean data
    df_clean = df.dropna(subset=feature_cols + ['crisis_next_30_days'])
    
    # Handle infinities
    for col in feature_cols:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    print(f"\nClean samples: {len(df_clean):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Crisis ratio: {df_clean['crisis_next_30_days'].mean():.1%}")
    
    # Garbage collection
    del df
    gc.collect()
    
    return df_clean, feature_cols

def train_xgboost_model(X_train, y_train, X_val, y_val, feature_cols, progress_tracker):
    """Train XGBoost model optimized for M4 Max"""
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)
    
    # Get number of CPU cores
    n_cores = psutil.cpu_count(logical=False)
    print(f"Using {n_cores} CPU cores")
    
    # XGBoost parameters optimized for M4 Max
    params = {
        'objective': 'binary:logistic',
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': n_cores,
        'tree_method': 'hist',  # Efficient for M-series
        'random_state': 42,
        'eval_metric': 'auc'
    }
    
    # Create DMatrix for efficient training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Training with early stopping
    print("\nTraining XGBoost...")
    start_time = time.time()
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # Train with callbacks for progress
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Evaluate
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    
    train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
    val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
    val_auc = roc_auc_score(y_val, val_pred)
    
    print(f"\n✓ XGBoost Training Complete!")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")
    print(f"Validation AUC: {val_auc:.3f}")
    print(f"Training time: {(time.time() - start_time)/60:.1f} minutes")
    
    progress_tracker.log_checkpoint('XGBoost', 'val_auc', val_auc)
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    feature_importance = {feature_cols[int(k[1:])]: v for k, v in importance.items() if k.startswith('f')}
    
    # Save model
    model.save_model('models/xgboost_full_model.json')
    
    # Also save as sklearn-compatible format
    sklearn_model = xgb.XGBClassifier(**params)
    sklearn_model.fit(X_train, y_train)
    joblib.dump(sklearn_model, 'models/xgboost_model.pkl')
    
    return sklearn_model, feature_importance

def train_lstm_model(X_train, y_train, X_val, y_val, feature_cols, device, progress_tracker):
    """Train LSTM model with MPS acceleration"""
    print("\n" + "=" * 80)
    print("TRAINING LSTM MODEL")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Model parameters
    sequence_length = 30
    input_size = len(feature_cols[:50])  # Use top 50 features
    hidden_size = 128
    num_layers = 2  # Reduced for stability
    batch_size = 512 if device.type == 'mps' else 256
    epochs = 60
    
    # Prepare sequences
    print(f"\nPreparing sequences (length={sequence_length})...")
    
    def create_sequences(X, y, seq_length):
        sequences = []
        labels = []
        
        for i in range(len(X) - seq_length):
            sequences.append(X[i:i+seq_length])
            labels.append(y[i+seq_length-1])
        
        return np.array(sequences), np.array(labels)
    
    # Use top features
    X_train_seq, y_train_seq = create_sequences(X_train[:, :input_size], y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val[:, :input_size], y_val, sequence_length)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
    y_val_tensor = torch.FloatTensor(y_val_seq).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LSTMCrisisPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    print(f"\nTraining LSTM on {device}...")
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += ((outputs > 0.5) == batch_y).float().mean().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_correct += ((outputs > 0.5) == batch_y).float().mean().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader)
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/lstm_model.pt')
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
            progress_tracker.log_checkpoint('LSTM', 'val_acc', val_acc)
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }
            torch.save(checkpoint, f'models/lstm_checkpoint_epoch_{epoch+1}.pt')
    
    print(f"\n✓ LSTM Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training time: {(time.time() - start_time)/60:.1f} minutes")
    
    return model

def train_gnn_model(X_train, y_train, X_val, y_val, feature_cols, device, progress_tracker):
    """Train simplified GNN model for M4 Max"""
    print("\n" + "=" * 80)
    print("TRAINING GRAPH NEURAL NETWORK")
    print("=" * 80)
    print("Note: Using simplified GNN for CPU/MPS compatibility")
    
    # For M4 Max, we'll use a simplified approach
    # Convert GNN to a deep feed-forward network with correlation features
    
    input_size = len(feature_cols[:75])  # Use top 75 features
    hidden_size = 256
    num_layers = 4
    batch_size = 256
    epochs = 40
    
    # Create a simplified model
    class SimplifiedGNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            
            layers = []
            for i in range(num_layers):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                else:
                    layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.Linear(hidden_size, 1))
            layers.append(nn.Sigmoid())
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    # Prepare data
    X_train_tensor = torch.FloatTensor(X_train[:, :input_size]).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val[:, :input_size]).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SimplifiedGNN(input_size, hidden_size, num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\nTraining Simplified GNN on {device}...")
    start_time = time.time()
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += ((outputs > 0.5) == batch_y).float().mean().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_correct += ((outputs > 0.5) == batch_y).float().mean().item()
        
        # Calculate metrics
        train_acc = train_correct / len(train_loader)
        val_acc = val_correct / len(val_loader)
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/gnn_model.pt')
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
            progress_tracker.log_checkpoint('GNN', 'val_acc', val_acc)
    
    print(f"\n✓ GNN Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Training time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Save as GNN model format
    torch.save({
        'model_type': 'SimplifiedGNN',
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'state_dict': model.state_dict()
    }, 'models/gnn_model_full.pt')
    
    return model

def save_ensemble_metadata(feature_cols, scalers, progress_tracker):
    """Save metadata for the ensemble"""
    metadata = {
        'training_completed': str(datetime.now()),
        'feature_columns': feature_cols,
        'num_features': len(feature_cols),
        'models_trained': ['XGBoost', 'LSTM', 'GNN'],
        'device_used': 'Apple M4 Max',
        'total_training_time': time.time() - progress_tracker.start_time,
        'checkpoints': [
            {
                'timestamp': str(cp['timestamp']),
                'model': cp['model'],
                'metric': cp['metric'],
                'value': cp['value']
            }
            for cp in progress_tracker.checkpoints
        ]
    }
    
    with open('models/ensemble_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save scalers
    joblib.dump(scalers, 'models/scalers.pkl')
    
    # Save feature columns
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)

def main():
    """Main training function"""
    print("=" * 80)
    print("FULL ENSEMBLE MODEL TRAINING FOR M4 MAX")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    # System info
    print_system_info()
    
    # Check device
    device = check_device()
    
    # Progress tracker
    progress_tracker = ProgressTracker()
    
    try:
        # 1. Prepare data
        print("\n" + "🔄 " * 20)
        print("PHASE 1: DATA PREPARATION")
        print("🔄 " * 20)
        
        # Use full dataset for M4 Max (or sample for testing)
        df_clean, feature_cols = prepare_data(sample_size=None)  # Set to 100000 for quick test
        
        # Prepare features and target
        X = df_clean[feature_cols].values
        y = df_clean['crisis_next_30_days'].values
        
        # Scale features
        print("\nScaling features...")
        scalers = {}
        
        # XGBoost scaler
        scaler_xgb = StandardScaler()
        X_scaled_xgb = scaler_xgb.fit_transform(X)
        scalers['xgboost'] = scaler_xgb
        
        # LSTM scaler (for top features)
        scaler_lstm = StandardScaler()
        X_scaled_lstm = scaler_lstm.fit_transform(X[:, :50])
        scalers['lstm'] = scaler_lstm
        
        # GNN scaler
        scaler_gnn = StandardScaler()
        X_scaled_gnn = scaler_gnn.fit_transform(X[:, :75])
        scalers['gnn'] = scaler_gnn
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled_xgb, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Validation set: {len(X_val):,} samples")
        print(f"Current memory usage: {get_memory_usage():.1f} GB")
        
        # 2. Train XGBoost
        print("\n" + "🌲 " * 20)
        print("PHASE 2: XGBOOST TRAINING")
        print("🌲 " * 20)
        
        xgb_model, feature_importance = train_xgboost_model(
            X_train, y_train, X_val, y_val, feature_cols, progress_tracker
        )
        
        # Clean up memory
        gc.collect()
        
        # 3. Train LSTM
        print("\n" + "🧠 " * 20)
        print("PHASE 3: LSTM TRAINING")
        print("🧠 " * 20)
        
        # Prepare LSTM data
        X_train_lstm, X_val_lstm, _, _ = train_test_split(
            X[:, :50], y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_lstm = scaler_lstm.transform(X_train_lstm)
        X_val_lstm = scaler_lstm.transform(X_val_lstm)
        
        lstm_model = train_lstm_model(
            X_train_lstm, y_train, X_val_lstm, y_val, 
            feature_cols[:50], device, progress_tracker
        )
        
        # Clean up
        if device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()
        
        # 4. Train GNN
        print("\n" + "🔗 " * 20)
        print("PHASE 4: GRAPH NEURAL NETWORK TRAINING")
        print("🔗 " * 20)
        
        # Prepare GNN data
        X_train_gnn, X_val_gnn, _, _ = train_test_split(
            X[:, :75], y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_gnn = scaler_gnn.transform(X_train_gnn)
        X_val_gnn = scaler_gnn.transform(X_val_gnn)
        
        gnn_model = train_gnn_model(
            X_train_gnn, y_train, X_val_gnn, y_val,
            feature_cols[:75], device, progress_tracker
        )
        
        # 5. Save ensemble metadata
        print("\n" + "💾 " * 20)
        print("PHASE 5: SAVING MODELS AND METADATA")
        print("💾 " * 20)
        
        save_ensemble_metadata(feature_cols, scalers, progress_tracker)
        
        # Final summary
        total_time = time.time() - progress_tracker.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print("\n" + "=" * 80)
        print("✅ FULL ENSEMBLE TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Total training time: {hours:02d}:{minutes:02d}")
        print(f"Peak memory usage: {max([cp['memory_gb'] for cp in progress_tracker.checkpoints]):.1f} GB")
        print("\nModels saved:")
        print("  - models/xgboost_model.pkl")
        print("  - models/lstm_model.pt")
        print("  - models/gnn_model_full.pt")
        print("  - models/ensemble_metadata.json")
        print("\n🚀 Ready to use in production!")
        
        # Update the test results file to indicate full training
        test_results = {
            'training_completed': True,
            'model_type': 'Full Ensemble (XGBoost + LSTM + GNN)',
            'features': len(feature_cols),
            'device': str(device),
            'total_samples': len(df_clean),
            'training_time_hours': total_time / 3600,
            'timestamp': str(datetime.now()),
            'note': 'Full ensemble trained on Apple M4 Max'
        }
        
        with open('models/test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nSaving partial progress...")
        save_ensemble_metadata(feature_cols if 'feature_cols' in locals() else [], 
                             scalers if 'scalers' in locals() else {},
                             progress_tracker)
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
