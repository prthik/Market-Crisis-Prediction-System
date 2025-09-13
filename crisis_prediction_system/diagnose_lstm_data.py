"""
Diagnostic script to investigate LSTM data issues
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SYSTEM_CONFIG
from data_collector import HistoricalEventLabeler

def diagnose_data():
    """Diagnose the data issue"""
    print("=" * 80)
    print("LSTM DATA DIAGNOSTICS")
    print("=" * 80)
    
    # Load data
    data_path = SYSTEM_CONFIG['HISTORICAL_DATA_PATH']
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records")
    
    # Label historical events
    event_labeler = HistoricalEventLabeler(data_path)
    df = event_labeler.label_data(df)
    
    # Check is_crisis column
    print(f"\nis_crisis column stats:")
    print(f"  Total crisis records: {df['is_crisis'].sum():,}")
    print(f"  Crisis ratio: {df['is_crisis'].mean():.2%}")
    
    # Create target the same way as in training
    df['crisis_next_30_days'] = df.groupby('Ticker')['is_crisis'].transform(
        lambda x: x.rolling(window=30, min_periods=1).max().shift(-30)
    ).fillna(0).astype(int)
    
    print(f"\ncrisis_next_30_days column stats:")
    print(f"  Total crisis records: {df['crisis_next_30_days'].sum():,}")
    print(f"  Crisis ratio: {df['crisis_next_30_days'].mean():.2%}")
    
    # Check by ticker
    print(f"\nChecking crisis distribution by ticker:")
    ticker_crisis = df.groupby('Ticker')['crisis_next_30_days'].agg(['sum', 'count', 'mean'])
    ticker_crisis = ticker_crisis[ticker_crisis['sum'] > 0].sort_values('sum', ascending=False)
    
    print(f"Tickers with crisis labels: {len(ticker_crisis)}")
    print("\nTop 10 tickers with most crisis labels:")
    print(ticker_crisis.head(10))
    
    # Check date range of crisis
    crisis_dates = df[df['crisis_next_30_days'] == 1]['Date']
    if len(crisis_dates) > 0:
        print(f"\nCrisis date range:")
        print(f"  First crisis: {crisis_dates.min()}")
        print(f"  Last crisis: {crisis_dates.max()}")
    
    # Simulate sequence creation for one ticker
    print("\n" + "=" * 80)
    print("SIMULATING SEQUENCE CREATION")
    print("=" * 80)
    
    # Pick a ticker with crisis
    if len(ticker_crisis) > 0:
        test_ticker = ticker_crisis.index[0]
        print(f"\nTesting with ticker: {test_ticker}")
        
        ticker_data = df[df['Ticker'] == test_ticker].sort_values('Date')
        print(f"Ticker data length: {len(ticker_data)}")
        print(f"Crisis records: {ticker_data['crisis_next_30_days'].sum()}")
        
        # Create sequences
        sequence_length = 60
        y = ticker_data['crisis_next_30_days'].values
        
        print(f"\nCreating sequences with length {sequence_length}...")
        crisis_sequences = 0
        
        for i in range(len(y) - sequence_length + 1):
            label = y[i + sequence_length - 1]
            if label == 1:
                crisis_sequences += 1
        
        print(f"Crisis sequences: {crisis_sequences}")
        
        # Check specific indices
        crisis_indices = np.where(y == 1)[0]
        print(f"\nCrisis indices in original data: {crisis_indices[:10]}...")
        
        # Check if crisis indices fall within sequence range
        valid_crisis = [idx for idx in crisis_indices if idx >= sequence_length - 1]
        print(f"Crisis indices that would be in sequences: {len(valid_crisis)}")

if __name__ == "__main__":
    diagnose_data()
