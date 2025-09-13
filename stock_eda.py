import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockDataExplorer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.parquet_file = self.dataset_path / "all_stock_data.parquet"
        self.csv_file = self.dataset_path / "all_stock_data.csv"
        
    def get_basic_info(self):
        """Get basic information about the dataset files"""
        print("=== DATASET OVERVIEW ===")
        print(f"Dataset location: {self.dataset_path}")
        
        if self.parquet_file.exists():
            parquet_size = self.parquet_file.stat().st_size / (1024**3)  # GB
            print(f"Parquet file size: {parquet_size:.2f} GB")
        
        if self.csv_file.exists():
            csv_size = self.csv_file.stat().st_size / (1024**3)  # GB
            print(f"CSV file size: {csv_size:.2f} GB")
    
    def sample_data_structure(self, n_rows=1000):
        """Safely examine data structure using a small sample"""
        print("\n=== DATA STRUCTURE ANALYSIS ===")
        
        try:
            # Use parquet for faster loading
            if self.parquet_file.exists():
                print("Loading sample from parquet file...")
                df_sample = pd.read_parquet(self.parquet_file, engine='pyarrow').head(n_rows)
            else:
                print("Loading sample from CSV file...")
                df_sample = pd.read_csv(self.csv_file, nrows=n_rows)
            
            print(f"\nSample data shape: {df_sample.shape}")
            print(f"Columns: {list(df_sample.columns)}")
            print(f"Data types:\n{df_sample.dtypes}")
            
            print(f"\nFirst few rows:")
            print(df_sample.head())
            
            print(f"\nBasic statistics:")
            print(df_sample.describe())
            
            return df_sample
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def analyze_tickers(self, sample_df):
        """Analyze ticker symbols in the dataset"""
        print("\n=== TICKER ANALYSIS ===")
        
        if 'ticker' in sample_df.columns:
            ticker_col = 'ticker'
        elif 'symbol' in sample_df.columns:
            ticker_col = 'symbol'
        elif 'Ticker' in sample_df.columns:
            ticker_col = 'Ticker'
        else:
            # Try to identify ticker column
            for col in sample_df.columns:
                if sample_df[col].dtype == 'object' and sample_df[col].str.len().max() <= 10:
                    ticker_col = col
                    break
            else:
                print("Could not identify ticker column")
                return
        
        print(f"Using column '{ticker_col}' as ticker symbol")
        unique_tickers = sample_df[ticker_col].nunique()
        print(f"Unique tickers in sample: {unique_tickers}")
        print(f"Sample tickers: {sample_df[ticker_col].unique()[:20]}")
        
        # Ticker frequency
        ticker_counts = sample_df[ticker_col].value_counts().head(10)
        print(f"\nTop 10 most frequent tickers in sample:")
        print(ticker_counts)
        
        return ticker_col
    
    def analyze_date_range(self, sample_df):
        """Analyze date range in the dataset"""
        print("\n=== DATE RANGE ANALYSIS ===")
        
        # Find date column
        date_col = None
        for col in sample_df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            print("Could not identify date column")
            return None
        
        print(f"Using column '{date_col}' as date")
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(sample_df[date_col]):
            sample_df[date_col] = pd.to_datetime(sample_df[date_col])
        
        print(f"Date range in sample: {sample_df[date_col].min()} to {sample_df[date_col].max()}")
        print(f"Total days in sample: {(sample_df[date_col].max() - sample_df[date_col].min()).days}")
        
        return date_col
    
    def analyze_price_columns(self, sample_df):
        """Analyze price-related columns"""
        print("\n=== PRICE ANALYSIS ===")
        
        # Identify price columns
        price_cols = []
        for col in sample_df.columns:
            if any(price_term in col.lower() for price_term in ['open', 'high', 'low', 'close', 'price', 'adj']):
                if sample_df[col].dtype in ['float64', 'int64']:
                    price_cols.append(col)
        
        print(f"Identified price columns: {price_cols}")
        
        if price_cols:
            print(f"\nPrice statistics:")
            print(sample_df[price_cols].describe())
            
            # Check for missing values
            print(f"\nMissing values in price columns:")
            for col in price_cols:
                missing_pct = (sample_df[col].isnull().sum() / len(sample_df)) * 100
                print(f"{col}: {missing_pct:.2f}%")
        
        return price_cols
    
    def analyze_volume(self, sample_df):
        """Analyze volume data"""
        print("\n=== VOLUME ANALYSIS ===")
        
        volume_col = None
        for col in sample_df.columns:
            if 'volume' in col.lower():
                volume_col = col
                break
        
        if volume_col:
            print(f"Volume column: {volume_col}")
            print(f"Volume statistics:")
            print(sample_df[volume_col].describe())
            
            # Check for zero volume days
            zero_volume_pct = (sample_df[volume_col] == 0).sum() / len(sample_df) * 100
            print(f"Zero volume days: {zero_volume_pct:.2f}%")
        else:
            print("No volume column found")
        
        return volume_col
    
    def create_sample_visualizations(self, sample_df, ticker_col, date_col, price_cols, volume_col):
        """Create basic visualizations from sample data"""
        print("\n=== CREATING SAMPLE VISUALIZATIONS ===")
        
        # Create output directory
        os.makedirs('eda_plots', exist_ok=True)
        
        # 1. Price distribution
        if price_cols:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(price_cols[:4], 1):  # Limit to 4 price columns
                plt.subplot(2, 2, i)
                plt.hist(sample_df[col].dropna(), bins=50, alpha=0.7)
                plt.title(f'{col} Distribution')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig('eda_plots/price_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Price distribution plots saved")
        
        # 2. Sample time series for a few tickers
        if ticker_col and date_col and price_cols:
            top_tickers = sample_df[ticker_col].value_counts().head(3).index
            
            plt.figure(figsize=(15, 10))
            for i, ticker in enumerate(top_tickers, 1):
                ticker_data = sample_df[sample_df[ticker_col] == ticker].sort_values(date_col)
                if len(ticker_data) > 1 and price_cols:
                    plt.subplot(3, 1, i)
                    close_col = next((col for col in price_cols if 'close' in col.lower()), price_cols[0])
                    plt.plot(ticker_data[date_col], ticker_data[close_col], marker='o', markersize=3)
                    plt.title(f'{ticker} - {close_col} Price')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('eda_plots/sample_time_series.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Sample time series plots saved")
        
        # 3. Volume analysis
        if volume_col:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(sample_df[volume_col].dropna(), bins=50, alpha=0.7)
            plt.title('Volume Distribution')
            plt.xlabel('Volume')
            plt.ylabel('Frequency')
            plt.yscale('log')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(sample_df[volume_col].dropna())
            plt.title('Volume Box Plot')
            plt.ylabel('Volume')
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig('eda_plots/volume_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Volume analysis plots saved")
    
    def get_full_dataset_info(self):
        """Get information about the full dataset without loading it entirely"""
        print("\n=== FULL DATASET INFORMATION ===")
        
        try:
            if self.parquet_file.exists():
                # Use parquet metadata to get info without loading full dataset
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(self.parquet_file)
                
                print(f"Total rows: {parquet_file.metadata.num_rows:,}")
                print(f"Number of row groups: {parquet_file.metadata.num_row_groups}")
                print(f"Schema: {parquet_file.schema}")
                
                # Get approximate date range by reading first and last chunks
                first_chunk = pd.read_parquet(self.parquet_file, engine='pyarrow').head(1000)
                last_chunk = pd.read_parquet(self.parquet_file, engine='pyarrow').tail(1000)
                
                # Find date column
                date_col = None
                for col in first_chunk.columns:
                    if 'date' in col.lower():
                        date_col = col
                        break
                
                if date_col:
                    first_chunk[date_col] = pd.to_datetime(first_chunk[date_col])
                    last_chunk[date_col] = pd.to_datetime(last_chunk[date_col])
                    
                    print(f"Approximate date range: {first_chunk[date_col].min()} to {last_chunk[date_col].max()}")
                
        except Exception as e:
            print(f"Could not get full dataset info: {e}")
    
    def run_full_analysis(self):
        """Run complete exploratory data analysis"""
        print("Starting Stock Market Data Exploratory Analysis")
        print("=" * 60)
        
        # Basic info
        self.get_basic_info()
        
        # Sample analysis
        sample_df = self.sample_data_structure(n_rows=5000)  # Larger sample for better analysis
        
        if sample_df is not None:
            ticker_col = self.analyze_tickers(sample_df)
            date_col = self.analyze_date_range(sample_df)
            price_cols = self.analyze_price_columns(sample_df)
            volume_col = self.analyze_volume(sample_df)
            
            # Create visualizations
            self.create_sample_visualizations(sample_df, ticker_col, date_col, price_cols, volume_col)
            
            # Full dataset info
            self.get_full_dataset_info()
            
            print("\n" + "=" * 60)
            print("EXPLORATORY DATA ANALYSIS COMPLETE")
            print("Check the 'eda_plots' directory for visualizations")
            print("=" * 60)
            
            return {
                'sample_df': sample_df,
                'ticker_col': ticker_col,
                'date_col': date_col,
                'price_cols': price_cols,
                'volume_col': volume_col
            }
        
        return None

if __name__ == "__main__":
    # Path to the downloaded dataset
    dataset_path = "/Users/abhinavagarwal/.cache/kagglehub/datasets/jakewright/9000-tickers-of-stock-market-data-full-history/versions/2"
    
    # Create explorer and run analysis
    explorer = StockDataExplorer(dataset_path)
    results = explorer.run_full_analysis()
