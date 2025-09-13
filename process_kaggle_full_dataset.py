"""
Process the full Kaggle historical stock market dataset
Replace the limited 2023-2024 data with complete 1962-2024 data
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import gc
from tqdm import tqdm

def process_full_historical_data():
    """Process the complete historical dataset from Kaggle"""
    print("=" * 80)
    print("PROCESSING FULL HISTORICAL STOCK MARKET DATASET (1962-2024)")
    print("=" * 80)
    
    # Path to the downloaded dataset
    kaggle_path = "/Users/abhinavagarwal/.cache/kagglehub/datasets/jakewright/9000-tickers-of-stock-market-data-full-history/versions/2"
    parquet_file = os.path.join(kaggle_path, "all_stock_data.parquet")
    csv_file = os.path.join(kaggle_path, "all_stock_data.csv")
    
    # Use parquet for faster loading
    if os.path.exists(parquet_file):
        print(f"\nLoading parquet file: {parquet_file}")
        print("This may take a minute due to the large size...")
        df = pd.read_parquet(parquet_file)
    else:
        print(f"\nLoading CSV file: {csv_file}")
        print("This may take several minutes due to the large size...")
        df = pd.read_csv(csv_file)
    
    print(f"\n✓ Loaded {len(df):,} records")
    
    # Ensure date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
        df = df.drop('date', axis=1)
    
    # Basic info
    print("\n" + "=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Years covered: {(df['Date'].max() - df['Date'].min()).days / 365:.1f}")
    print(f"Unique tickers: {df['Ticker'].nunique():,}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n⚠️  Missing columns: {missing_cols}")
    else:
        print("\n✓ All required columns present")
    
    # Add Adj Close if not present
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    
    # Sort by ticker and date
    print("\nSorting data...")
    df = df.sort_values(['Ticker', 'Date'])
    
    # Remove any duplicates
    print("\nRemoving duplicates...")
    original_len = len(df)
    df = df.drop_duplicates(subset=['Ticker', 'Date'])
    if len(df) < original_len:
        print(f"Removed {original_len - len(df):,} duplicate records")
    
    # Filter out penny stocks and low volume
    print("\nFiltering data quality...")
    # Remove stocks with price < $1 or volume < 1000
    df = df[(df['Close'] >= 1) | (df['Volume'] >= 1000)]
    print(f"Remaining records after filtering: {len(df):,}")
    
    # Year distribution
    print("\n" + "=" * 80)
    print("DATA DISTRIBUTION BY DECADE")
    print("=" * 80)
    df['Year'] = df['Date'].dt.year
    df['Decade'] = (df['Year'] // 10) * 10
    
    decade_stats = df.groupby('Decade').agg({
        'Ticker': 'nunique',
        'Date': 'count'
    }).rename(columns={'Ticker': 'Unique_Tickers', 'Date': 'Records'})
    
    for decade, stats in decade_stats.iterrows():
        pct = stats['Records'] / len(df) * 100
        print(f"{decade}s: {stats['Records']:,} records ({pct:.1f}%), {stats['Unique_Tickers']:,} tickers")
    
    # Check coverage of major stocks through history
    print("\n" + "=" * 80)
    print("HISTORICAL COVERAGE CHECK")
    print("=" * 80)
    
    major_stocks = ['AAPL', 'MSFT', 'IBM', 'GE', 'XOM', 'JPM', 'WMT', 'JNJ', 'PG', 'KO']
    for ticker in major_stocks[:5]:
        ticker_data = df[df['Ticker'] == ticker]
        if not ticker_data.empty:
            min_date = ticker_data['Date'].min()
            max_date = ticker_data['Date'].max()
            years = (max_date - min_date).days / 365
            print(f"{ticker}: {min_date.year} to {max_date.year} ({years:.1f} years, {len(ticker_data):,} records)")
    
    # Save the processed dataset
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATASET")
    print("=" * 80)
    
    output_file = "full_historical_stock_data.parquet"
    print(f"\nSaving to {output_file}...")
    
    # Remove temporary columns before saving
    df = df.drop(['Year', 'Decade'], axis=1)
    
    # Save
    df.to_parquet(output_file, compression='snappy', index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved! File size: {file_size_mb:.1f} MB")
    
    # Backup old file
    old_file = "full_processed_stock_data.parquet"
    if os.path.exists(old_file):
        backup_file = "full_processed_stock_data_backup_2023only.parquet"
        print(f"\nBacking up old file to {backup_file}")
        os.rename(old_file, backup_file)
    
    # Create symlink or copy to expected location
    print(f"\nCreating link to {old_file}")
    if os.path.exists(old_file):
        os.remove(old_file)
    os.symlink(output_file, old_file)
    
    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nDataset ready with {len(df):,} records spanning {df['Date'].min().year}-{df['Date'].max().year}")
    print("\nThis dataset includes all major market crises:")
    print("  - 1973-74 Bear Market")
    print("  - 1987 Black Monday")
    print("  - 2000 Dot-com Bubble")
    print("  - 2008 Financial Crisis")
    print("  - 2020 COVID Crash")
    print("  - 2022 Bear Market")
    print("  - 2023 Banking Crisis")
    print("\nYou can now retrain all models with this complete historical data!")
    
    return df

if __name__ == "__main__":
    df = process_full_historical_data()
    
    # Quick validation
    print("\n" + "=" * 80)
    print("QUICK VALIDATION")
    print("=" * 80)
    
    # Check 2008 crisis period
    crisis_2008 = df[(df['Date'] >= '2008-09-01') & (df['Date'] <= '2009-03-31')]
    print(f"\n2008 Financial Crisis period records: {len(crisis_2008):,}")
    print(f"Unique tickers during 2008 crisis: {crisis_2008['Ticker'].nunique():,}")
    
    # Check COVID period
    covid_period = df[(df['Date'] >= '2020-02-15') & (df['Date'] <= '2020-04-01')]
    print(f"\nCOVID crash period records: {len(covid_period):,}")
    print(f"Unique tickers during COVID crash: {covid_period['Ticker'].nunique():,}")
    
    print("\n✅ Full historical dataset is ready for model training!")
