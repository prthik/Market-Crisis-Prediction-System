"""
Download the FULL historical stock market dataset from Kaggle
This should get us data going back many years, not just 2023-2024
"""
import kagglehub
import os
import pandas as pd
import numpy as np
from datetime import datetime

def download_full_dataset():
    """Download the complete historical stock market dataset from Kaggle"""
    print("=" * 80)
    print("DOWNLOADING FULL HISTORICAL STOCK MARKET DATASET")
    print("=" * 80)
    
    # Download the dataset
    print("\nDownloading from Kaggle...")
    print("Dataset: jakewright/9000-tickers-of-stock-market-data-full-history")
    print("This may take several minutes depending on your internet connection...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("jakewright/9000-tickers-of-stock-market-data-full-history")
        print(f"\n✓ Download complete!")
        print(f"Path to dataset files: {path}")
        
        # List all files in the directory
        print("\n" + "=" * 80)
        print("DATASET FILES")
        print("=" * 80)
        
        files = os.listdir(path)
        print(f"Found {len(files)} files:")
        
        csv_files = [f for f in files if f.endswith('.csv')]
        parquet_files = [f for f in files if f.endswith('.parquet')]
        
        if csv_files:
            print(f"\nCSV files ({len(csv_files)}):")
            for f in csv_files[:10]:  # Show first 10
                file_path = os.path.join(path, f)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {f} ({size_mb:.1f} MB)")
            if len(csv_files) > 10:
                print(f"  ... and {len(csv_files) - 10} more")
        
        if parquet_files:
            print(f"\nParquet files ({len(parquet_files)}):")
            for f in parquet_files:
                file_path = os.path.join(path, f)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {f} ({size_mb:.1f} MB)")
        
        # Check a sample file to verify date range
        print("\n" + "=" * 80)
        print("VERIFYING DATA RANGE")
        print("=" * 80)
        
        sample_file = None
        if csv_files:
            # Find a well-known stock
            for ticker in ['AAPL.csv', 'MSFT.csv', 'IBM.csv', 'GE.csv']:
                if ticker in csv_files:
                    sample_file = ticker
                    break
            
            if not sample_file:
                sample_file = csv_files[0]
            
            print(f"\nChecking {sample_file}...")
            sample_path = os.path.join(path, sample_file)
            
            # Read sample
            df_sample = pd.read_csv(sample_path, nrows=100000)
            
            # Convert date column
            if 'Date' in df_sample.columns:
                df_sample['Date'] = pd.to_datetime(df_sample['Date'])
            elif 'date' in df_sample.columns:
                df_sample['Date'] = pd.to_datetime(df_sample['date'])
            
            print(f"Columns: {list(df_sample.columns)}")
            print(f"Shape: {df_sample.shape}")
            
            if 'Date' in df_sample.columns:
                # Read full file to get complete date range
                print(f"\nReading full file to check complete date range...")
                df_full = pd.read_csv(sample_path)
                df_full['Date'] = pd.to_datetime(df_full['Date'] if 'Date' in df_full.columns else df_full['date'])
                
                min_date = df_full['Date'].min()
                max_date = df_full['Date'].max()
                
                print(f"\nDate range for {sample_file}:")
                print(f"  Earliest: {min_date}")
                print(f"  Latest: {max_date}")
                print(f"  Years covered: {(max_date - min_date).days / 365:.1f}")
                print(f"  Total records: {len(df_full):,}")
                
                # Check for historical data
                historical_years = df_full['Date'].dt.year.unique()
                print(f"\nYears present: {sorted(historical_years)}")
                
                if min_date.year < 2000:
                    print("✓ EXCELLENT! This dataset contains historical data going back decades!")
                elif min_date.year < 2010:
                    print("✓ GOOD! This dataset contains data from before 2010")
                else:
                    print("⚠️ WARNING: This dataset might also be limited in historical range")
        
        # Create a script to process the full dataset
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Run process_kaggle_full_dataset.py to combine all ticker files")
        print("2. This will create a comprehensive dataset with all historical data")
        print("3. Then retrain the models with the complete historical data")
        
        return path
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have kagglehub installed: pip install kagglehub")
        print("2. Ensure you're authenticated with Kaggle")
        print("3. Check your internet connection")
        return None

if __name__ == "__main__":
    dataset_path = download_full_dataset()
    
    if dataset_path:
        print(f"\n✅ Dataset downloaded to: {dataset_path}")
        print("\nYou can now process this data to create a comprehensive training dataset")
