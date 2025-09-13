import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class StockDataProcessor:
    """
    Comprehensive data processor for stock market data with feature engineering
    capabilities for technical indicators and preparation for news integration.
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.parquet_file = self.dataset_path / "all_stock_data.parquet"
        self.processed_data = None
        
    def load_data_chunk(self, start_date=None, end_date=None, tickers=None, chunk_size=100000):
        """
        Load data in chunks to manage memory efficiently
        """
        print(f"Loading data chunk...")
        
        try:
            # Load full dataset (we'll optimize this for large-scale processing later)
            if start_date or end_date or tickers:
                # For now, load a sample for development
                df = pd.read_parquet(self.parquet_file, engine='pyarrow')
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter by date range
                if start_date:
                    df = df[df['Date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['Date'] <= pd.to_datetime(end_date)]
                
                # Filter by tickers
                if tickers:
                    df = df[df['Ticker'].isin(tickers)]
                    
                print(f"Loaded {len(df):,} records")
                return df
            else:
                # Load recent data for development (last 2 years)
                df = pd.read_parquet(self.parquet_file, engine='pyarrow')
                df['Date'] = pd.to_datetime(df['Date'])
                recent_date = df['Date'].max() - pd.Timedelta(days=730)  # 2 years
                df = df[df['Date'] >= recent_date]
                print(f"Loaded recent data: {len(df):,} records")
                return df
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """
        Clean and preprocess the raw stock data
        """
        print("Cleaning data...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Sort by ticker and date
        df_clean = df_clean.sort_values(['Ticker', 'Date'])
        
        # Handle Open = 0.0 cases
        # Replace with previous close or average of High/Low
        mask_zero_open = df_clean['Open'] == 0.0
        
        # Method 1: Use previous close
        df_clean['Prev_Close'] = df_clean.groupby('Ticker')['Close'].shift(1)
        df_clean.loc[mask_zero_open, 'Open'] = df_clean.loc[mask_zero_open, 'Prev_Close']
        
        # Method 2: For remaining zeros, use average of High/Low
        still_zero = df_clean['Open'] == 0.0
        df_clean.loc[still_zero, 'Open'] = (df_clean.loc[still_zero, 'High'] + 
                                           df_clean.loc[still_zero, 'Low']) / 2
        
        # Remove the temporary column
        df_clean = df_clean.drop('Prev_Close', axis=1)
        
        # Handle any remaining missing values
        df_clean = df_clean.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Ensure volume is non-negative
        df_clean['Volume'] = df_clean['Volume'].clip(lower=0)
        
        # Basic data validation
        # Ensure High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        invalid_hloc = (
            (df_clean['High'] < df_clean['Low']) |
            (df_clean['High'] < df_clean['Open']) |
            (df_clean['High'] < df_clean['Close']) |
            (df_clean['Low'] > df_clean['Open']) |
            (df_clean['Low'] > df_clean['Close'])
        )
        
        if invalid_hloc.sum() > 0:
            print(f"Warning: Found {invalid_hloc.sum()} records with invalid HLOC relationships")
            # Remove invalid records
            df_clean = df_clean[~invalid_hloc]
        
        print(f"Data cleaned. Shape: {df_clean.shape}")
        return df_clean
    
    def calculate_technical_indicators(self, df):
        """
        Calculate comprehensive technical indicators
        """
        print("Calculating technical indicators...")
        
        df_tech = df.copy()
        
        # Initialize new columns with NaN
        new_columns = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position', 'Volume_SMA_10', 'Volume_SMA_20',
            'Volume_Ratio', 'OBV', 'True_Range', 'ATR', 'Returns', 'Volatility_20', 'RSI', 'Stoch_K', 'Stoch_D',
            'ROC_10', 'Williams_R'
        ]
        
        for col in new_columns:
            df_tech[col] = np.nan
        
        # Group by ticker for calculations
        for ticker in df_tech['Ticker'].unique():
            mask = df_tech['Ticker'] == ticker
            ticker_data = df_tech[mask].copy()
            
            if len(ticker_data) < 50:  # Skip tickers with insufficient data
                continue
            
            # Reset index for easier processing
            ticker_data = ticker_data.reset_index(drop=True)
            
            # Price-based indicators
            ticker_data = self._add_price_indicators(ticker_data)
            
            # Volume-based indicators
            ticker_data = self._add_volume_indicators(ticker_data)
            
            # Volatility indicators
            ticker_data = self._add_volatility_indicators(ticker_data)
            
            # Momentum indicators
            ticker_data = self._add_momentum_indicators(ticker_data)
            
            # Update the main dataframe
            df_tech.loc[mask, ticker_data.columns] = ticker_data.values
        
        print(f"Technical indicators calculated. New shape: {df_tech.shape}")
        return df_tech
    
    def _add_price_indicators(self, df):
        """Add price-based technical indicators"""
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    def _add_volume_indicators(self, df):
        """Add volume-based indicators"""
        
        # Volume Moving Averages
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Volume Ratio
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # On-Balance Volume (OBV)
        df['Price_Change'] = df['Close'].diff()
        df['OBV'] = 0
        
        for i in range(1, len(df)):
            if df.iloc[i]['Price_Change'] > 0:
                df.iloc[i, df.columns.get_loc('OBV')] = df.iloc[i-1]['OBV'] + df.iloc[i]['Volume']
            elif df.iloc[i]['Price_Change'] < 0:
                df.iloc[i, df.columns.get_loc('OBV')] = df.iloc[i-1]['OBV'] - df.iloc[i]['Volume']
            else:
                df.iloc[i, df.columns.get_loc('OBV')] = df.iloc[i-1]['OBV']
        
        df = df.drop('Price_Change', axis=1)
        return df
    
    def _add_volatility_indicators(self, df):
        """Add volatility-based indicators"""
        
        # True Range
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close_Prev'] = abs(df['High'] - df['Close'].shift(1))
        df['Low_Close_Prev'] = abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = df[['High_Low', 'High_Close_Prev', 'Low_Close_Prev']].max(axis=1)
        
        # Average True Range (ATR)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # Historical Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Clean up temporary columns
        df = df.drop(['High_Low', 'High_Close_Prev', 'Low_Close_Prev'], axis=1)
        
        return df
    
    def _add_momentum_indicators(self, df):
        """Add momentum-based indicators"""
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Rate of Change (ROC)
        df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        return df
    
    def create_features_for_prediction(self, df):
        """
        Create features specifically designed for prediction models
        """
        print("Creating prediction features...")
        
        df_features = df.copy()
        
        # Lag features (previous day values)
        lag_columns = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Position']
        for col in lag_columns:
            if col in df_features.columns:
                for lag in [1, 2, 3, 5]:
                    df_features[f'{col}_lag_{lag}'] = df_features.groupby('Ticker')[col].shift(lag)
        
        # Price ratios and differences
        df_features['Close_to_SMA_20'] = df_features['Close'] / df_features['SMA_20']
        df_features['Close_to_SMA_50'] = df_features['Close'] / df_features['SMA_50']
        df_features['High_Low_Ratio'] = df_features['High'] / df_features['Low']
        df_features['Open_Close_Ratio'] = df_features['Open'] / df_features['Close']
        
        # Volatility features
        df_features['Price_Range'] = (df_features['High'] - df_features['Low']) / df_features['Close']
        df_features['Gap'] = (df_features['Open'] - df_features['Close'].shift(1)) / df_features['Close'].shift(1)
        
        # Target variables (next day predictions)
        df_features['Next_Close'] = df_features.groupby('Ticker')['Close'].shift(-1)
        df_features['Next_Return'] = (df_features['Next_Close'] - df_features['Close']) / df_features['Close']
        df_features['Next_Direction'] = (df_features['Next_Return'] > 0).astype(int)
        
        # Remove rows with insufficient data
        df_features = df_features.dropna()
        
        print(f"Prediction features created. Final shape: {df_features.shape}")
        return df_features
    
    def prepare_news_integration_features(self, df):
        """
        Prepare features for news/events integration
        """
        print("Preparing news integration features...")
        
        df_news = df.copy()
        
        # Add time-based features for news alignment
        df_news['Year'] = df_news['Date'].dt.year
        df_news['Month'] = df_news['Date'].dt.month
        df_news['DayOfWeek'] = df_news['Date'].dt.dayofweek
        df_news['Quarter'] = df_news['Date'].dt.quarter
        
        # Market regime indicators (bull/bear market proxies)
        df_news['SMA_200'] = df_news.groupby('Ticker')['Close'].rolling(window=200).mean().reset_index(0, drop=True)
        df_news['Above_SMA_200'] = (df_news['Close'] > df_news['SMA_200']).astype(int)
        
        # Volatility regime
        df_news['High_Volatility'] = (df_news['Volatility_20'] > df_news['Volatility_20'].rolling(window=50).mean()).astype(int)
        
        # Placeholder columns for news features (to be populated later)
        df_news['News_Sentiment'] = 0.0  # Will be filled with sentiment scores
        df_news['News_Volume'] = 0.0     # Number of news articles
        df_news['Event_Type'] = 'none'   # Type of major events
        
        print(f"News integration features prepared. Shape: {df_news.shape}")
        return df_news
    
    def process_data_pipeline(self, start_date=None, end_date=None, tickers=None):
        """
        Complete data processing pipeline
        """
        print("Starting complete data processing pipeline...")
        print("=" * 50)
        
        # Step 1: Load data
        df = self.load_data_chunk(start_date, end_date, tickers)
        if df is None:
            return None
        
        # Step 2: Clean data
        df = self.clean_data(df)
        
        # Step 3: Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Step 4: Create prediction features
        df = self.create_features_for_prediction(df)
        
        # Step 5: Prepare for news integration
        df = self.prepare_news_integration_features(df)
        
        self.processed_data = df
        
        print("=" * 50)
        print("Data processing pipeline completed!")
        print(f"Final dataset shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Number of tickers: {df['Ticker'].nunique()}")
        
        return df
    
    def save_processed_data(self, filename="processed_stock_data.parquet"):
        """
        Save processed data to file
        """
        if self.processed_data is not None:
            self.processed_data.to_parquet(filename, engine='pyarrow', index=False)
            print(f"Processed data saved to {filename}")
        else:
            print("No processed data to save. Run process_data_pipeline first.")

if __name__ == "__main__":
    # Example usage
    dataset_path = "/Users/abhinavagarwal/.cache/kagglehub/datasets/jakewright/9000-tickers-of-stock-market-data-full-history/versions/2"
    
    # Initialize processor
    processor = StockDataProcessor(dataset_path)
    
    # Process data for a subset of popular tickers
    popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    # Process recent data (last 2 years) for these tickers
    processed_df = processor.process_data_pipeline(
        start_date='2022-01-01',
        tickers=popular_tickers
    )
    
    if processed_df is not None:
        # Save processed data
        processor.save_processed_data("processed_stock_features.parquet")
        
        # Display sample of processed features
        print("\nSample of processed features:")
        print(processed_df[['Date', 'Ticker', 'Close', 'RSI', 'MACD', 'BB_Position', 'Next_Return']].head(10))
