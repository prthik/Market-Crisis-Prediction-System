"""
LSTM Training Script for M4 Max - Full Dataset
Trains LSTM model with MPS acceleration on entire historical dataset
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
from models import LSTMCrisisPredictor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.Process().memory_info().rss / 1024 / 1024 / 1024

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
        print("⚠️ Using CPU")
    return device

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data"""
    def __init__(self, X, y, sequence_length=60, stride=1):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Calculate valid indices
        self.indices = []
        for i in range(0, len(X) - sequence_length + 1, stride):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        X_seq = self.X[start_idx:end_idx]
        y_seq = self.y[end_idx - 1]  # Predict at the last timestep
        
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_seq])

def prepare_lstm_data():
    """Prepare data specifically for LSTM training"""
    print("=" * 80)
    print("PREPARING FULL DATASET FOR LSTM")
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
    
    # Sort by ticker and date for proper sequences
    df = df.sort_values(['Ticker', 'Date'])
    
    # Create simplified features for LSTM (memory efficient)
    print("\nCreating LSTM-specific features...")
    
    # Price features
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df.groupby('Ticker')['Close'].shift(1))
    
    # Price relative to moving averages
    df['SMA_20'] = df.groupby('Ticker')['Close'].rolling(20, min_periods=5).mean().reset_index(0, drop=True)
    df['SMA_50'] = df.groupby('Ticker')['Close'].rolling(50, min_periods=10).mean().reset_index(0, drop=True)
    df['price_to_SMA20'] = df['Close'] / df['SMA_20']
    df['price_to_SMA50'] = df['Close'] / df['SMA_50']
    
    # Volatility
    df['volatility_20'] = df.groupby('Ticker')['returns'].rolling(20, min_periods=5).std().reset_index(0, drop=True) * np.sqrt(252)
    df['volatility_60'] = df.groupby('Ticker')['returns'].rolling(60, min_periods=10).std().reset_index(0, drop=True) * np.sqrt(252)
    
    # Volume
    df['volume_SMA_20'] = df.groupby('Ticker')['Volume'].rolling(20, min_periods=5).mean().reset_index(0, drop=True)
    df['volume_ratio'] = df['Volume'] / df['volume_SMA_20']
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))
    
    # High/Low features
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_to_high'] = df['Close'] / df['High']
    df['close_to_low'] = df['Close'] / df['Low']
    
    # Market features
    df['market_volatility'] = df.groupby('Date')['returns'].transform('std') * np.sqrt(252)
    df['market_returns'] = df.groupby('Date')['returns'].transform('mean')
    
    # Create target
    df['crisis_next_30_days'] = df.groupby('Ticker')['is_crisis'].transform(
        lambda x: x.rolling(window=30, min_periods=1).max().shift(-30)
    ).fillna(0).astype(int)
    
    # Select features for LSTM
    feature_cols = [
        'returns', 'log_returns', 'price_to_SMA20', 'price_to_SMA50',
        'volatility_20', 'volatility_60', 'volume_ratio', 'RSI',
        'high_low_ratio', 'close_to_high', 'close_to_low',
        'market_volatility', 'market_returns'
    ]
    
    print(f"Selected {len(feature_cols)} features for LSTM")
    
    # Clean data
    df_clean = df[['Ticker', 'Date'] + feature_cols + ['crisis_next_30_days']].copy()
    
    # Handle infinities and NaN
    for col in feature_cols:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        df_clean[col] = df_clean[col].fillna(0)  # Simple fill for LSTM
    
    df_clean = df_clean.dropna()
    
    print(f"\nClean samples: {len(df_clean):,}")
    print(f"Crisis ratio: {df_clean['crisis_next_30_days'].mean():.2%}")
    print(f"Memory usage: {get_memory_usage():.1f} GB")
    
    return df_clean, feature_cols

def create_sequences_by_ticker(df, feature_cols, sequence_length=60):
    """Create sequences grouped by ticker"""
    print(f"\nCreating sequences (length={sequence_length})...")
    
    all_sequences = []
    all_labels = []
    
    # Process each ticker separately
    tickers = df['Ticker'].unique()
    
    for ticker in tqdm(tickers, desc="Processing tickers"):
        ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
        
        if len(ticker_data) < sequence_length:
            continue
        
        # Extract features
        X = ticker_data[feature_cols].values
        y = ticker_data['crisis_next_30_days'].values
        
        # Create sequences for this ticker
        for i in range(len(X) - sequence_length + 1):
            all_sequences.append(X[i:i+sequence_length])
            all_labels.append(y[i+sequence_length-1])
    
    return np.array(all_sequences), np.array(all_labels)

def train_lstm_full(X_sequences, y_sequences, feature_cols, device):
    """Train LSTM on full dataset"""
    print("\n" + "=" * 80)
    print("TRAINING LSTM MODEL")
    print("=" * 80)
    
    # Split data
    print("Splitting data...")
    
    # Ensure y_sequences is integer type
    y_sequences_int = y_sequences.astype(int)
    
    # Check if we have any positive samples
    print(f"Total positive samples: {y_sequences_int.sum():,} ({y_sequences_int.mean():.2%})")
    
    if y_sequences_int.sum() < 100:
        print("WARNING: Very few positive samples. Using random split instead of stratified.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_sequences, y_sequences_int, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_sequences, y_sequences_int, test_size=0.2, random_state=42, stratify=y_sequences_int
        )
    
    print(f"Training sequences: {len(X_train):,}")
    print(f"Validation sequences: {len(X_val):,}")
    print(f"Positive class ratio - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")
    
    # Model parameters
    input_size = len(feature_cols)
    hidden_size = MODEL_CONFIG['LSTM']['hidden_size']
    num_layers = MODEL_CONFIG['LSTM']['num_layers']
    dropout = MODEL_CONFIG['LSTM']['dropout']
    learning_rate = MODEL_CONFIG['LSTM']['learning_rate']
    
    # Batch size optimized for M4 Max unified memory
    batch_size = 512 if device.type == 'mps' else 256
    epochs = 100
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Important for MPS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = LSTMCrisisPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # Calculate positive weight for class imbalance (MPS requires float32)
    if y_train.mean() > 0:
        pos_weight_value = float(1/y_train.mean() - 1)
    else:
        pos_weight_value = 1.0
        print("WARNING: No positive samples in training set!")
    
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    print(f"\nTraining on {device}...")
    start_time = time.time()
    best_val_auc = 0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        from sklearn.metrics import roc_auc_score
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), 'models/lstm_model.pt')
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            print(f"  Best Val AUC: {best_val_auc:.4f} (epoch {best_epoch+1})")
            print(f"  Memory: {get_memory_usage():.1f} GB")
        
        # Early stopping
        if epoch - best_epoch > 20:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    
    # Save metadata
    metadata = {
        'model': 'LSTM',
        'training_date': str(datetime.now()),
        'sequence_length': X_sequences.shape[1],
        'n_features': len(feature_cols),
        'training_sequences': len(X_train),
        'validation_sequences': len(X_val),
        'training_time_minutes': training_time / 60,
        'performance': {
            'best_val_auc': float(best_val_auc),
            'best_epoch': best_epoch + 1,
            'final_train_auc': float(history['train_auc'][-1]),
            'final_val_auc': float(history['val_auc'][-1])
        },
        'architecture': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        },
        'device': str(device),
        'history': history
    }
    
    with open('models/lstm_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ LSTM model saved successfully!")
    
    return model, history

def main():
    """Main training function"""
    print("=" * 80)
    print("LSTM TRAINING ON FULL DATASET - M4 MAX")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print(f"Initial memory usage: {get_memory_usage():.1f} GB")
    
    # Check device
    device = check_device()
    
    try:
        # Prepare data
        df_clean, feature_cols = prepare_lstm_data()
        
        # Create sequences
        sequence_length = MODEL_CONFIG['LSTM']['sequence_length']
        X_sequences, y_sequences = create_sequences_by_ticker(
            df_clean, feature_cols, sequence_length
        )
        
        print(f"\nCreated {len(X_sequences):,} sequences")
        print(f"Sequence shape: {X_sequences.shape}")
        
        # Save scaler (using same features as sequences)
        scaler = StandardScaler()
        # Fit on flattened sequences
        X_flat = X_sequences.reshape(-1, X_sequences.shape[-1])
        scaler.fit(X_flat)
        
        # Scale sequences
        X_sequences_scaled = X_sequences.copy()
        for i in range(len(X_sequences)):
            X_sequences_scaled[i] = scaler.transform(X_sequences[i])
        
        # Save scaler
        scalers = joblib.load('models/scalers.pkl')
        scalers['lstm'] = scaler
        joblib.dump(scalers, 'models/scalers.pkl')
        
        # Save LSTM feature columns
        with open('models/lstm_features.json', 'w') as f:
            json.dump(feature_cols, f)
        
        # Train model
        model, history = train_lstm_full(X_sequences_scaled, y_sequences, feature_cols, device)
        
        print("\n" + "=" * 80)
        print("✅ LSTM TRAINING COMPLETE!")
        print("=" * 80)
        print("Model saved to: models/lstm_model.pt")
        print("Metadata: models/lstm_metadata.json")
        
        # Clean up GPU memory
        if device.type == 'mps':
            torch.mps.empty_cache()
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now()}")
    print(f"Final memory usage: {get_memory_usage():.1f} GB")

if __name__ == "__main__":
    main()
