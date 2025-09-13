"""
Feature engineering module for crisis prediction
Creates advanced features from market data for prediction models
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CrisisFeatureEngineer:
    """Engineers features for crisis prediction"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df.groupby('Ticker')['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df.groupby('Ticker')['Close'].rolling(period).mean().reset_index(0, drop=True)
            df[f'price_to_SMA_{period}'] = df['Close'] / df[f'SMA_{period}']
        
        # Exponential moving averages
        for span in [12, 26]:
            df[f'EMA_{span}'] = df.groupby('Ticker')['Close'].ewm(span=span).mean().reset_index(0, drop=True)
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df.groupby('Ticker')['MACD'].ewm(span=9).mean().reset_index(0, drop=True)
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI'] = df.groupby('Ticker')['Close'].apply(lambda x: calculate_rsi(x)).reset_index(0, drop=True)
        
        # Bollinger Bands
        df['BB_middle'] = df.groupby('Ticker')['Close'].rolling(20).mean().reset_index(0, drop=True)
        bb_std = df.groupby('Ticker')['Close'].rolling(20).std().reset_index(0, drop=True)
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume features
        df['volume_SMA_20'] = df.groupby('Ticker')['Volume'].rolling(20).mean().reset_index(0, drop=True)
        df['volume_ratio'] = df['Volume'] / df['volume_SMA_20']
        
        # Price patterns
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        df = df.copy()
        
        # Historical volatility
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}'] = df.groupby('Ticker')['returns'].rolling(period).std().reset_index(0, drop=True) * np.sqrt(252)
        
        # Volatility ratios
        df['vol_ratio_20_60'] = df['volatility_20'] / df['volatility_60']
        df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
        
        # GARCH-like features
        df['squared_returns'] = df['returns'] ** 2
        df['vol_clustering'] = df.groupby('Ticker')['squared_returns'].rolling(20).mean().reset_index(0, drop=True)
        
        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(np.log(df['High'] / df['Low']) ** 2 / (4 * np.log(2))) * np.sqrt(252)
        
        # Average True Range (ATR)
        df['TR'] = df[['High', 'Low', 'Close']].apply(
            lambda x: max(x['High'] - x['Low'], 
                         abs(x['High'] - x['Close']), 
                         abs(x['Low'] - x['Close'])), axis=1)
        df['ATR'] = df.groupby('Ticker')['TR'].rolling(14).mean().reset_index(0, drop=True)
        
        return df
    
    def create_correlation_features(self, df: pd.DataFrame, market_index: str = 'SPY') -> pd.DataFrame:
        """Create correlation-based features"""
        df = df.copy()
        
        # Get market returns
        market_returns = df[df['Ticker'] == market_index][['Date', 'returns']].rename(columns={'returns': 'market_returns'})
        df = df.merge(market_returns, on='Date', how='left')
        
        # Rolling correlations with market
        for period in [20, 60]:
            df[f'corr_market_{period}'] = df.groupby('Ticker').apply(
                lambda x: x['returns'].rolling(period).corr(x['market_returns'])
            ).reset_index(0, drop=True)
        
        # Beta calculation
        df['beta_60'] = df.groupby('Ticker').apply(
            lambda x: x['returns'].rolling(60).cov(x['market_returns']) / x['market_returns'].rolling(60).var()
        ).reset_index(0, drop=True)
        
        return df
    
    def create_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        df = df.copy()
        
        # Spread proxies
        df['HL_spread'] = (df['High'] - df['Low']) / df['Close']
        df['CO_spread'] = abs(df['Close'] - df['Open']) / df['Close']
        
        # Liquidity proxies
        df['amihud_illiquidity'] = abs(df['returns']) / (df['Volume'] * df['Close'])
        df['turnover'] = df['Volume'] / df['Volume'].rolling(252).mean()
        
        # Price efficiency
        df['price_efficiency'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
        
        return df
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime features"""
        df = df.copy()
        
        # Bull/Bear market indicator
        df['above_SMA_200'] = (df['Close'] > df['SMA_200']).astype(int)
        df['below_SMA_200'] = (df['Close'] <= df['SMA_200']).astype(int)
        
        # Volatility regime
        vol_median = df['volatility_20'].rolling(252).median()
        df['high_vol_regime'] = (df['volatility_20'] > vol_median * 1.5).astype(int)
        
        # Trend strength
        df['trend_strength'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']
        
        # Market breadth (if we have multiple stocks)
        if 'Date' in df.columns:
            breadth = df.groupby('Date')['above_SMA_200'].mean()
            df = df.merge(breadth.rename('market_breadth'), on='Date', how='left')
        
        return df
    
    def create_extreme_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for extreme event detection"""
        df = df.copy()
        
        # Tail risk measures
        df['left_tail_5'] = df.groupby('Ticker')['returns'].rolling(252).apply(
            lambda x: np.percentile(x, 5)
        ).reset_index(0, drop=True)
        
        df['right_tail_95'] = df.groupby('Ticker')['returns'].rolling(252).apply(
            lambda x: np.percentile(x, 95)
        ).reset_index(0, drop=True)
        
        # Extreme return indicators
        df['extreme_negative'] = (df['returns'] < df['left_tail_5']).astype(int)
        df['extreme_positive'] = (df['returns'] > df['right_tail_95']).astype(int)
        
        # Cumulative extreme events
        df['cum_extreme_neg_20'] = df.groupby('Ticker')['extreme_negative'].rolling(20).sum().reset_index(0, drop=True)
        
        # Jump detection
        df['price_jump'] = (abs(df['returns']) > df['volatility_20'] * 3).astype(int)
        
        return df
    
    def create_cross_asset_features(self, df: pd.DataFrame, asset_data: Dict) -> pd.DataFrame:
        """Create features from multiple asset classes"""
        df = df.copy()
        
        # Add VIX if available
        if 'vix' in asset_data:
            df['VIX'] = asset_data['vix']
            df['VIX_change'] = df['VIX'].pct_change()
            df['VIX_high'] = (df['VIX'] > 30).astype(int)
        
        # Add economic indicators
        if 'economic_indicators' in asset_data:
            indicators = asset_data['economic_indicators']
            if 'Yield Curve' in indicators:
                df['yield_curve'] = indicators['Yield Curve']
                df['inverted_yield_curve'] = (df['yield_curve'] < 0).astype(int)
        
        # Add currency data
        if 'currency_data' in asset_data:
            currency = asset_data['currency_data']
            if 'DXY_proxy' in currency:
                df['dollar_strength'] = currency['DXY_proxy']
                df['dollar_change'] = df['dollar_strength'].pct_change()
        
        return df
    
    def create_sentiment_features(self, df: pd.DataFrame, sentiment_data: Dict) -> pd.DataFrame:
        """Create sentiment-based features"""
        df = df.copy()
        
        if sentiment_data:
            df['news_sentiment'] = sentiment_data.get('average_sentiment', 0)
            df['sentiment_volatility'] = sentiment_data.get('sentiment_std', 0)
            df['negative_news_ratio'] = sentiment_data.get('negative_ratio', 0)
            df['news_volume'] = sentiment_data.get('article_count', 0)
            
            # Sentiment extremes
            df['extreme_negative_sentiment'] = (df['news_sentiment'] < -0.5).astype(int)
            df['sentiment_shock'] = abs(df['news_sentiment'].diff()) > 0.3
        
        return df
    
    def create_crisis_proximity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on similarity to historical crisis patterns"""
        df = df.copy()
        
        # Crisis pattern score (combination of multiple indicators)
        crisis_indicators = []
        
        # Add indicators that exist
        if 'volatility_20' in df.columns and 'volatility_60' in df.columns:
            crisis_indicators.append((df['volatility_20'] > df['volatility_60'] * 1.5))  # Volatility spike
        
        if 'corr_market_20' in df.columns:
            crisis_indicators.append((df['corr_market_20'] > 0.8))  # High correlation
            
        if 'cum_extreme_neg_20' in df.columns:
            crisis_indicators.append((df['cum_extreme_neg_20'] > 3))  # Multiple extreme events
            
        if 'below_SMA_200' in df.columns:
            crisis_indicators.append((df['below_SMA_200'] == 1))  # Below long-term average
            
        if 'volume_ratio' in df.columns:
            crisis_indicators.append((df['volume_ratio'] > 2))  # Volume spike
            
        if 'VIX_high' in df.columns:
            crisis_indicators.append((df['VIX_high'] == 1))  # High VIX
        
        if crisis_indicators:
            df['crisis_score'] = sum(indicator.astype(int) for indicator in crisis_indicators) / len(crisis_indicators)
        else:
            df['crisis_score'] = 0
        
        # Rolling crisis score
        df['crisis_score_avg_5'] = df['crisis_score'].rolling(5).mean()
        df['crisis_score_max_20'] = df['crisis_score'].rolling(20).max()
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, asset_data: Dict = None, 
                            sentiment_data: Dict = None) -> pd.DataFrame:
        """Create all features for crisis prediction"""
        print("Engineering comprehensive feature set...")
        
        # Technical features
        df = self.create_technical_features(df)
        print("✓ Technical features created")
        
        # Volatility features
        df = self.create_volatility_features(df)
        print("✓ Volatility features created")
        
        # Correlation features
        if 'SPY' in df['Ticker'].unique():
            df = self.create_correlation_features(df)
            print("✓ Correlation features created")
        
        # Market structure features
        df = self.create_market_structure_features(df)
        print("✓ Market structure features created")
        
        # Regime features
        df = self.create_regime_features(df)
        print("✓ Regime features created")
        
        # Extreme event features
        df = self.create_extreme_event_features(df)
        print("✓ Extreme event features created")
        
        # Cross-asset features
        if asset_data:
            df = self.create_cross_asset_features(df, asset_data)
            print("✓ Cross-asset features created")
        
        # Sentiment features
        if sentiment_data:
            df = self.create_sentiment_features(df, sentiment_data)
            print("✓ Sentiment features created")
        
        # Crisis proximity features
        df = self.create_crisis_proximity_features(df)
        print("✓ Crisis proximity features created")
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in 
                            ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"\nTotal features created: {len(self.feature_names)}")
        
        return df
    
    def select_top_features(self, df: pd.DataFrame, target: str, n_features: int = 50) -> List[str]:
        """Select most important features based on correlation with target"""
        # Calculate correlations
        correlations = df[self.feature_names].corrwith(df[target]).abs()
        
        # Remove features with too many NaN values
        nan_ratio = df[self.feature_names].isna().sum() / len(df)
        valid_features = nan_ratio[nan_ratio < 0.3].index
        
        # Get top features
        top_features = correlations[valid_features].nlargest(n_features).index.tolist()
        
        return top_features
