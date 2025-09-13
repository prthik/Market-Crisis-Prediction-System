import pandas as pd
import numpy as np
from pathlib import Path
from advanced_visualizations import AdvancedStockVisualizer
import warnings
import gc
warnings.filterwarnings('ignore')

def load_complete_historical_dataset():
    """
    Load the complete historical dataset (34.6M records) in chunks
    """
    print("=" * 80)
    print("LOADING COMPLETE HISTORICAL STOCK DATASET (1962-2024)")
    print("=" * 80)
    
    dataset_path = Path("/Users/abhinavagarwal/.cache/kagglehub/datasets/jakewright/9000-tickers-of-stock-market-data-full-history/versions/2")
    parquet_file = dataset_path / "all_stock_data.parquet"
    
    print("\n1. Loading the COMPLETE dataset (34.6 million records)...")
    print("   This will take several minutes due to the data size...")
    
    try:
        # Load the entire dataset
        df = pd.read_parquet(parquet_file, engine='pyarrow')
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"\n✓ Successfully loaded complete dataset!")
        print(f"   Total records: {len(df):,}")
        print(f"   Unique tickers: {df['Ticker'].nunique():,}")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        
        return df
    except Exception as e:
        print(f"Error loading complete dataset: {e}")
        return None

def process_historical_data_efficiently(df):
    """
    Process the historical data with efficient feature engineering
    """
    print("\n" + "=" * 80)
    print("PROCESSING HISTORICAL DATA WITH FEATURE ENGINEERING")
    print("=" * 80)
    
    # Sort by ticker and date
    print("\n2. Sorting data by ticker and date...")
    df = df.sort_values(['Ticker', 'Date'])
    
    # Clean data
    print("\n3. Cleaning data...")
    # Remove invalid OHLC relationships
    invalid_hloc = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) |
        (df['High'] < df['Close']) |
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close'])
    )
    df = df[~invalid_hloc]
    
    # Handle zero/missing values
    df = df[df['Open'] > 0]
    df = df[df['Volume'] >= 0]
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    print(f"   After cleaning: {len(df):,} records")
    
    # Add basic features needed for visualization
    print("\n4. Adding essential features...")
    
    # Calculate returns
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Calculate simple moving averages (for volatility analysis)
    print("   Calculating moving averages...")
    df['SMA_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['SMA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50, min_periods=1).mean())
    
    # Calculate volatility
    print("   Calculating volatility...")
    df['Volatility_20'] = df.groupby('Ticker')['Daily_Return'].transform(
        lambda x: x.rolling(20, min_periods=1).std() * np.sqrt(252)
    )
    
    # Volume metrics
    print("   Calculating volume metrics...")
    df['Volume_SMA_20'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # Basic technical indicators
    print("   Adding basic technical indicators...")
    
    # RSI calculation (simplified)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))
    
    # Market regime indicators
    df['Above_SMA_50'] = (df['Close'] > df['SMA_50']).astype(int)
    
    print(f"\n✓ Feature engineering completed!")
    print(f"   Final shape: {df.shape}")
    
    return df

def save_complete_dataset(df, filename="complete_historical_stock_data.parquet"):
    """
    Save the complete processed dataset
    """
    print(f"\n5. Saving complete dataset to {filename}...")
    df.to_parquet(filename, engine='pyarrow', index=False)
    print(f"   ✓ Saved successfully!")
    
    # Display summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total records: {len(df):,}")
    print(f"Unique tickers: {df['Ticker'].nunique():,}")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Years covered: {(df['Date'].max() - df['Date'].min()).days / 365:.1f}")
    
    # Show ticker distribution
    print("\nTop 20 stocks by data availability:")
    print(df['Ticker'].value_counts().head(20))
    
    # Show date distribution by decade
    print("\nRecords by decade:")
    df['Decade'] = (df['Date'].dt.year // 10) * 10
    decade_counts = df.groupby('Decade').size()
    for decade, count in decade_counts.items():
        print(f"  {decade}s: {count:,} records")

def regenerate_visualizations_complete(data_file="complete_historical_stock_data.parquet"):
    """
    Regenerate all visualizations with the complete historical dataset
    """
    print("\n" + "=" * 80)
    print("REGENERATING VISUALIZATIONS WITH COMPLETE HISTORICAL DATA")
    print("=" * 80)
    
    # Initialize visualizer with complete dataset
    visualizer = AdvancedStockVisualizer(data_file)
    
    if visualizer.load_data():
        print(f"\nLoaded complete historical dataset for visualization")
        
        # Run all visualizations
        results = visualizer.run_all_visualizations()
        
        print("\n✓ All visualizations regenerated with complete historical data!")
        return results
    else:
        print("Failed to load data for visualization")
        return None

if __name__ == "__main__":
    print("COMPREHENSIVE HISTORICAL STOCK MARKET ANALYSIS")
    print("Processing 34.6 million records from 1962-2024")
    print("This will take approximately 5-15 minutes depending on your system")
    print("\n" + "=" * 80)
    
    # Step 1: Load complete historical dataset
    df = load_complete_historical_dataset()
    
    if df is not None:
        # Step 2: Process with feature engineering
        df_processed = process_historical_data_efficiently(df)
        
        # Free up memory
        del df
        gc.collect()
        
        # Step 3: Save complete dataset
        save_complete_dataset(df_processed)
        
        # Step 4: Regenerate visualizations
        viz_results = regenerate_visualizations_complete()
        
        print("\n" + "=" * 80)
        print("COMPLETE HISTORICAL ANALYSIS FINISHED!")
        print("=" * 80)
        print("\nYou now have:")
        print("1. Complete processed dataset with 34.6M records (1962-2024)")
        print("2. Advanced visualizations showing patterns across 62+ years")
        print("3. Insights from all 9,315 stocks across all market conditions")
        print("\nThis comprehensive analysis provides the foundation for")
        print("building truly robust prediction models that have learned from")
        print("6+ decades of market behavior!")
    else:
        print("\nFailed to load the historical dataset.")
