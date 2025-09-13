"""
Improved LSTM Training Script for M4 Max - Full Historical Dataset
Fixes all issues: proper labeling, forward-looking predictions, MPS compatibility
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
from sklearn.metrics import roc_auc_score, classification_report

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

def prepare_lstm_data_improved():
    """Prepare data for LSTM with improved labeling"""
    print("=" * 80)
    print("PREPARING FULL HISTORICAL DATASET FOR LSTM")
    print("=" * 80)
    
    # Load data
    data_path = SYSTEM_CONFIG['HISTORICAL_DATA_PATH']
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Memory usage: {get_memory_usage():.1f} GB")
    
    # Label historical events
    event_labeler = HistoricalEventLabeler(data_path)
    df = event_labeler.label_data(df)
    crisis_count = df['is_crisis'].sum()
    print(f"✓ Historical events labeled: {crisis_count:,} crisis records ({crisis_count/len(df)*100:.1%})")
    
    # Sort by ticker and date for proper sequences
    df = df.sort_values(['Ticker', 'Date'])
    
    # Create LSTM-specific features
    print("\nCreating LSTM features...")
    
    # Price features
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df.groupby('Ticker')['Close'].shift(1))
    
    # Moving averages
    df['SMA_10'] = df.groupby('Ticker')['Close'].rolling(10, min_periods=5).mean().reset_index(0, drop=True)
    df['SMA_20'] = df.groupby('Ticker')['Close'].rolling(20, min_periods=10).mean().reset_index(0, drop=True)
    df['SMA_50'] = df.groupby('Ticker')['Close'].rolling(50, min_periods=20).mean().reset_index(0, drop=True)
    
    # Price ratios
    df['price_to_SMA10'] = df['Close'] / df['SMA_10']
    df['price_to_SMA20'] = df['Close'] / df['SMA_20']
    df['price_to_SMA50'] = df['Close'] / df['SMA_50']
    
    # Volatility
    df['volatility_10'] = df.groupby('Ticker')['returns'].rolling(10, min_periods=5).std().reset_index(0, drop=True) * np.sqrt(252)
    df['volatility_20'] = df.groupby('Ticker')['returns'].rolling(20, min_periods=10).std().reset_index(0, drop=True) * np.sqrt(252)
    
    # Volume features
    df['volume_SMA_20'] = df.groupby('Ticker')['Volume'].rolling(20, min_periods=10).mean().reset_index(0, drop=True)
    df['volume_ratio'] = df['Volume'] / df['volume_SMA_20']
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))
    
    # Price range features
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_to_high'] = df['Close'] / df['High']
    
    # Market-wide features
    df['market_returns'] = df.groupby('Date')['returns'].transform('mean')
    df['market_volatility'] = df.groupby('Date')['returns'].transform('std') * np.sqrt(252)
    
    # IMPROVED TARGET: Crisis in next 30 days (forward-looking)
    print("\nCreating forward-looking target...")
    df['crisis_next_30_days'] = 0
    
    # For each row, check if there's a crisis in the next 30 trading days
    for ticker in tqdm(df['Ticker'].unique(), desc="Creating targets"):
        ticker_mask = df['Ticker'] == ticker
        ticker_df = df[ticker_mask].copy()
        
        # Get crisis periods for this ticker
        crisis_mask = ticker_df['is_crisis'] == 1
        
        # For each point, look ahead 30 days
        for i in range(len(ticker_df) - 30):
            if ticker_df.iloc[i:i+30]['is_crisis'].max() == 1:
                df.loc[ticker_df.index[i], 'crisis_next_30_days'] = 1
    
    crisis_target_count = df['crisis_next_30_days'].sum()
    print(f"✓ Target created: {crisis_target_count:,} records predict crisis ({crisis_target_count/len(df)*100:.1%})")
    
    # Select features
    feature_cols = [
        'returns', 'log_returns', 
        'price_to_SMA10', 'price_to_SMA20', 'price_to_SMA50',
        'volatility_10', 'volatility_20', 
        'volume_ratio', 'RSI',
        'high_low_ratio', 'close_to_high',
        'market_returns', 'market_volatility'
    ]
    
    print(f"\nSelected {len(feature_cols)} features for LSTM")
    
    # Clean data
    df_clean = df[['Ticker', 'Date'] + feature_cols + ['crisis_next_30_days']].copy()
    
    # Handle infinities and NaN
    for col in feature_cols:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        # Use forward fill then backward fill for NaN
        df_clean[col] = df_clean.groupby('Ticker')[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Final cleaning
    df_clean = df_clean.dropna()
    
    print(f"\nClean samples: {len(df_clean):,}")
    print(f"Crisis prediction ratio: {df_clean['crisis_next_30_days'].mean():.2%}")
    print(f"Memory usage: {get_memory_usage():.1f} GB")
    
    return df_clean, feature_cols

def create_sequences_improved(df, feature_cols, sequence_length=20):
    """Create sequences with improved strategy"""
    print(f"\nCreating sequences (length={sequence_length})...")
    
    all_sequences = []
    all_labels = []
    
    # Process each ticker separately
    tickers = df['Ticker'].unique()
    
    for ticker in tqdm(tickers, desc="Processing tickers"):
        ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
        
        # Need at least sequence_length + 30 days for forward looking
        if len(ticker_data) < sequence_length + 30:
            continue
        
        # Extract features and labels
        X = ticker_data[feature_cols].values.astype(np.float32)  # Ensure float32
        y = ticker_data['crisis_next_30_days'].values.astype(np.float32)
        
        # Create sequences
        for i in range(len(X) - sequence_length + 1):
            # Only create sequences where we have full forward-looking window
            if i + sequence_length <= len(y):
                seq_X = X[i:i+sequence_length]
                seq_y = y[i+sequence_length-1]  # Label at end of sequence
                
                all_sequences.append(seq_X)
                all_labels.append(seq_y)
    
    X_sequences = np.array(all_sequences, dtype=np.float32)
    y_sequences = np.array(all_labels, dtype=np.float32)
    
    print(f"Created {len(X_sequences):,} sequences")
    print(f"Positive samples: {y_sequences.sum():,} ({y_sequences.mean():.2%})")
    
    return X_sequences, y_sequences

def train_lstm_improved(X_sequences, y_sequences, feature_cols, device, sequence_length):
    """Train LSTM with all improvements"""
    print("\n" + "=" * 80)
    print("TRAINING IMPROVED LSTM MODEL")
    print("=" * 80)
    
    # Split data with stratification
    print("Splitting data...")
    
    # Ensure integer labels for stratification
    y_int = y_sequences.astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_int
    )
    
    print(f"Training sequences: {len(X_train):,}")
    print(f"Validation sequences: {len(X_val):,}")
    print(f"Positive ratio - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")
    
    # Verify we have positive samples
    if y_train.sum() == 0 or y_val.sum() == 0:
        raise ValueError("No positive samples in train or validation set!")
    
    # Model parameters
    input_size = len(feature_cols)
    hidden_size = 64  # Reduced for stability
    num_layers = 2
    dropout = 0.3
    learning_rate = 0.001
    
    # Batch size optimized for M4 Max
    batch_size = 256
    epochs = 50  # Reduced for faster iteration
    
    # Create datasets - ensure float32
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
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
    
    # Loss with class weights (MPS-compatible)
    pos_weight = (1 - y_train.mean()) / y_train.mean()
    print(f"  Positive weight: {pos_weight:.2f}")
    
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
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
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_X, batch_y in progress_bar:
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
            
            progress_bar.set_postfix({'loss': loss.item()})
        
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
            torch.save(model.state_dict(), 'models/lstm_model_improved.pt')
        
        # Print progress
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        print(f"  Best Val AUC: {best_val_auc:.4f} (epoch {best_epoch+1})")
        
        # Early stopping
        if epoch - best_epoch > 10:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('models/lstm_model_improved.pt'))
    model.eval()
    
    # Final evaluation
    with torch.no_grad():
        val_preds_final = []
        val_labels_final = []
        
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = torch.sigmoid(model(batch_X))
            val_preds_final.extend(outputs.cpu().numpy())
            val_labels_final.extend(batch_y.numpy())
    
    val_preds_final = np.array(val_preds_final).flatten()
    val_labels_final = np.array(val_labels_final).flatten()
    
    print("\nFinal Validation Classification Report:")
    print(classification_report(
        val_labels_final, 
        (val_preds_final > 0.5).astype(int),
        target_names=['No Crisis', 'Crisis']
    ))
    
    # Save metadata
    metadata = {
        'model': 'LSTM_Improved',
        'training_date': str(datetime.now()),
        'dataset_info': {
            'total_sequences': len(X_sequences),
            'positive_ratio': float(y_sequences.mean()),
            'sequence_length': sequence_length,
            'n_features': len(feature_cols)
        },
        'architecture': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        },
        'performance': {
            'best_val_auc': float(best_val_auc),
            'best_epoch': best_epoch + 1,
            'training_time_minutes': training_time / 60
        },
        'features': feature_cols,
        'device': str(device)
    }
    
    with open('models/lstm_improved_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy to main model file
    torch.save(model.state_dict(), 'models/lstm_model.pt')
    
    print("\n✓ LSTM model saved successfully!")
    
    return model, history

def main():
    """Main training function"""
    print("=" * 80)
    print("IMPROVED LSTM TRAINING ON FULL HISTORICAL DATASET - M4 MAX")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print(f"Initial memory usage: {get_memory_usage():.1f} GB")
    
    # Check device
    device = check_device()
    
    try:
        # Prepare data
        df_clean, feature_cols = prepare_lstm_data_improved()
        
        # Create sequences with shorter length
        sequence_length = 20  # Reduced from 60
        X_sequences, y_sequences = create_sequences_improved(
            df_clean, feature_cols, sequence_length
        )
        
        # Save scaler
        print("\nFitting and saving scaler...")
        scaler = StandardScaler()
        X_flat = X_sequences.reshape(-1, X_sequences.shape[-1])
        scaler.fit(X_flat)
        
        # Scale sequences
        X_sequences_scaled = X_sequences.copy()
        for i in range(len(X_sequences)):
            X_sequences_scaled[i] = scaler.transform(X_sequences[i])
        
        # Save scaler
        if os.path.exists('models/scalers.pkl'):
            scalers = joblib.load('models/scalers.pkl')
        else:
            scalers = {}
        scalers['lstm'] = scaler
        joblib.dump(scalers, 'models/scalers.pkl')
        
        # Save feature columns
        with open('models/lstm_features.json', 'w') as f:
            json.dump(feature_cols, f)
        
        # Train model
        model, history = train_lstm_improved(
            X_sequences_scaled, y_sequences, feature_cols, device, sequence_length
        )
        
        print("\n" + "=" * 80)
        print("✅ LSTM TRAINING COMPLETE!")
        print("=" * 80)
        print("Models saved:")
        print("  - models/lstm_model.pt (main model)")
        print("  - models/lstm_model_improved.pt (backup)")
        print("  - models/lstm_improved_metadata.json")
        
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
