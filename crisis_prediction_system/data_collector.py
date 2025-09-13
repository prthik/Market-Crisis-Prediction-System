"""
Real-time data collection module
Integrates with multiple APIs to gather market data, economic indicators, and news
"""
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import json
from config import API_KEYS, SYSTEM_CONFIG
import warnings
warnings.filterwarnings('ignore')

class MarketDataCollector:
    """Collects real-time market data from multiple sources"""
    
    def __init__(self):
        self.alpha_vantage_key = API_KEYS['ALPHA_VANTAGE']
        self.polygon_key = API_KEYS['POLYGON']
        self.fred_key = API_KEYS['FRED']
        self.news_api_key = API_KEYS['NEWS_API']
        self.openexchange_key = API_KEYS['OPENEXCHANGE']
        
    def get_market_data(self, symbols: List[str], source: str = 'yahoo') -> pd.DataFrame:
        """Get real-time market data for given symbols"""
        if source == 'yahoo':
            return self._get_yahoo_data(symbols)
        elif source == 'alpha_vantage':
            return self._get_alpha_vantage_data(symbols)
        elif source == 'polygon':
            return self._get_polygon_data(symbols)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def _get_yahoo_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get data from Yahoo Finance"""
        try:
            # Get latest data
            data = yf.download(symbols, period='1d', interval='1h', progress=False)
            
            # Get additional info
            info_data = []
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                info_data.append({
                    'symbol': symbol,
                    'market_cap': info.get('marketCap', 0),
                    'volume': info.get('volume', 0),
                    'beta': info.get('beta', 1),
                    'pe_ratio': info.get('forwardPE', 0)
                })
            
            return data, pd.DataFrame(info_data)
        except Exception as e:
            print(f"Error fetching Yahoo data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _get_alpha_vantage_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get data from Alpha Vantage"""
        all_data = []
        
        for symbol in symbols[:5]:  # Rate limit: 5 calls/minute
            try:
                # Get intraday data
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_INTRADAY',
                    'symbol': symbol,
                    'interval': '60min',
                    'apikey': self.alpha_vantage_key
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if 'Time Series (60min)' in data:
                    df = pd.DataFrame.from_dict(data['Time Series (60min)'], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df['symbol'] = symbol
                    all_data.append(df)
                
                time.sleep(12)  # Rate limit
                
            except Exception as e:
                print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def _get_polygon_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get data from Polygon.io"""
        all_data = []
        
        for symbol in symbols:
            try:
                # Get latest quote
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
                params = {'apiKey': self.polygon_key}
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if data['status'] == 'OK' and 'results' in data:
                    df = pd.DataFrame(data['results'])
                    df['symbol'] = symbol
                    all_data.append(df)
                
            except Exception as e:
                print(f"Error fetching Polygon data for {symbol}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def get_vix_data(self) -> float:
        """Get current VIX level"""
        try:
            vix = yf.Ticker('^VIX')
            return vix.history(period='1d')['Close'].iloc[-1]
        except:
            return None
    
    def get_economic_indicators(self) -> Dict:
        """Get economic indicators from FRED"""
        indicators = {}
        
        # Check if API key is available
        if not self.fred_key:
            print("FRED API key not configured, using mock economic data")
            return {
                '10-Year Treasury': 3.5,
                '2-Year Treasury': 2.5,
                'Yield Curve': 1.0,
                'Fed Funds Rate': 2.0,
                'Unemployment Rate': 3.8,
                'CPI': 250.0,
                'USD/EUR Exchange Rate': 1.1
            }
        
        series_ids = {
            'DGS10': '10-Year Treasury',
            'DGS2': '2-Year Treasury',
            'DFF': 'Fed Funds Rate',
            'UNRATE': 'Unemployment Rate',
            'CPIAUCSL': 'CPI',
            'DEXUSEU': 'USD/EUR Exchange Rate'
        }
        
        for series_id, name in series_ids.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_key,
                    'file_type': 'json',
                    'limit': 1,
                    'sort_order': 'desc'
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if 'observations' in data and data['observations']:
                    value = float(data['observations'][0]['value'])
                    indicators[name] = value
                
            except Exception as e:
                print(f"Error fetching FRED data for {series_id}: {e}")
        
        # Calculate yield curve
        if '10-Year Treasury' in indicators and '2-Year Treasury' in indicators:
            indicators['Yield Curve'] = indicators['10-Year Treasury'] - indicators['2-Year Treasury']
        
        return indicators
    
    def get_news_sentiment(self, query: str = "stock market crisis") -> Dict:
        """Get news sentiment analysis"""
        try:
            # Check if API key is available
            if not self.news_api_key:
                print("News API key not configured, using mock sentiment data")
                return {
                    'average_sentiment': 0.1,
                    'sentiment_std': 0.2,
                    'article_count': 50,
                    'negative_ratio': 0.3
                }
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'ok':
                articles = data['articles']
                
                # Simple sentiment analysis based on keywords
                negative_keywords = ['crash', 'crisis', 'collapse', 'fear', 'panic', 
                                   'recession', 'bear', 'decline', 'plunge', 'tumble']
                positive_keywords = ['rally', 'surge', 'gain', 'bull', 'recovery', 
                                   'growth', 'rise', 'boom', 'optimism']
                
                sentiment_scores = []
                for article in articles:
                    title = article.get('title') or ''
                    description = article.get('description') or ''
                    text = (title + ' ' + description).lower()
                    
                    neg_score = sum(1 for word in negative_keywords if word in text)
                    pos_score = sum(1 for word in positive_keywords if word in text)
                    
                    if neg_score + pos_score > 0:
                        sentiment = (pos_score - neg_score) / (pos_score + neg_score)
                    else:
                        sentiment = 0
                    
                    sentiment_scores.append(sentiment)
                
                return {
                    'average_sentiment': np.mean(sentiment_scores),
                    'sentiment_std': np.std(sentiment_scores),
                    'article_count': len(articles),
                    'negative_ratio': sum(1 for s in sentiment_scores if s < -0.3) / len(sentiment_scores)
                }
            
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            
        return {'average_sentiment': 0, 'sentiment_std': 0, 'article_count': 0, 'negative_ratio': 0}
    
    def get_currency_data(self) -> Dict:
        """Get currency exchange rates"""
        try:
            # Check if API key is available
            if not self.openexchange_key:
                print("OpenExchange API key not configured, using mock currency data")
                return {
                    'USD_EUR': 0.92,
                    'USD_JPY': 145.0,
                    'USD_GBP': 0.79,
                    'DXY_proxy': 1.02
                }
            
            url = "https://openexchangerates.org/api/latest.json"
            params = {'app_id': self.openexchange_key}
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'rates' in data:
                # Calculate dollar strength index proxy
                major_currencies = ['EUR', 'JPY', 'GBP', 'CAD', 'CHF']
                dxy_proxy = 0
                
                for currency in major_currencies:
                    if currency in data['rates']:
                        dxy_proxy += 1 / data['rates'][currency]
                
                return {
                    'USD_EUR': 1 / data['rates']['EUR'],
                    'USD_JPY': data['rates']['JPY'],
                    'USD_GBP': 1 / data['rates']['GBP'],
                    'DXY_proxy': dxy_proxy / len(major_currencies)
                }
                
        except Exception as e:
            print(f"Error fetching currency data: {e}")
            
        return {
            'USD_EUR': 0.92,
            'USD_JPY': 145.0,
            'USD_GBP': 0.79,
            'DXY_proxy': 1.02
        }
    
    def get_sector_performance(self) -> pd.DataFrame:
        """Get sector ETF performance"""
        sector_etfs = {
            'XLF': 'Financials',
            'XLK': 'Technology',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        try:
            data = yf.download(list(sector_etfs.keys()), period='5d', progress=False)
            
            # Calculate 5-day returns
            returns = {}
            for etf in sector_etfs.keys():
                if 'Close' in data.columns:
                    prices = data['Close'][etf] if len(sector_etfs) > 1 else data['Close']
                    returns[sector_etfs[etf]] = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
            
            return pd.DataFrame.from_dict(returns, orient='index', columns=['5_day_return'])
            
        except Exception as e:
            print(f"Error fetching sector data: {e}")
            return pd.DataFrame()
    
    def collect_all_data(self) -> Dict:
        """Collect all available data"""
        print("Collecting comprehensive market data...")
        
        # Major indices and stocks (using ^VIX for VIX index)
        symbols = ['SPY', 'QQQ', 'DIA', 'IWM', '^VIX', 
                  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                  'JPM', 'BAC', 'GS', 'XOM', 'CVX']
        
        # Collect data from multiple sources
        market_data, market_info = self.get_market_data(symbols, source='yahoo')
        vix = self.get_vix_data()
        economic_indicators = self.get_economic_indicators()
        news_sentiment = self.get_news_sentiment()
        currency_data = self.get_currency_data()
        sector_performance = self.get_sector_performance()
        
        return {
            'market_data': market_data,
            'market_info': market_info,
            'vix': vix,
            'economic_indicators': economic_indicators,
            'news_sentiment': news_sentiment,
            'currency_data': currency_data,
            'sector_performance': sector_performance,
            'timestamp': datetime.now()
        }


class HistoricalEventLabeler:
    """Labels historical market events for model training"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.events = self._define_historical_events()
        
    def _define_historical_events(self) -> List[Dict]:
        """Define all historical market events"""
        events = [
            # 1960s-1970s
            {'start': '1962-05-28', 'end': '1962-06-26', 'type': 'MARKET_CRASH', 'name': 'Kennedy Slide', 'severity': 0.7},
            {'start': '1973-10-01', 'end': '1974-10-01', 'type': 'COMMODITY_SHOCK', 'name': 'Oil Crisis', 'severity': 0.85},
            {'start': '1973-01-11', 'end': '1974-12-06', 'type': 'MARKET_CRASH', 'name': '1973-74 Bear Market', 'severity': 0.9},
            
            # 1980s
            {'start': '1987-10-19', 'end': '1987-10-19', 'type': 'MARKET_CRASH', 'name': 'Black Monday', 'severity': 0.95},
            {'start': '1982-08-01', 'end': '1982-11-01', 'type': 'CURRENCY_CRISIS', 'name': 'Mexican Debt Crisis', 'severity': 0.7},
            {'start': '1989-01-01', 'end': '1991-12-31', 'type': 'FINANCIAL_CRISIS', 'name': 'S&L Crisis', 'severity': 0.8},
            
            # 1990s
            {'start': '1997-07-02', 'end': '1998-01-01', 'type': 'CURRENCY_CRISIS', 'name': 'Asian Financial Crisis', 'severity': 0.85},
            {'start': '1998-08-17', 'end': '1998-10-01', 'type': 'FINANCIAL_CRISIS', 'name': 'Russian Default/LTCM', 'severity': 0.8},
            
            # 2000s
            {'start': '2000-03-10', 'end': '2002-10-09', 'type': 'SECTOR_BUBBLE', 'name': 'Dot-com Bubble', 'severity': 0.9},
            {'start': '2001-09-11', 'end': '2001-09-21', 'type': 'GEOPOLITICAL_SHOCK', 'name': '9/11 Attacks', 'severity': 0.8},
            {'start': '2007-10-09', 'end': '2009-03-09', 'type': 'FINANCIAL_CRISIS', 'name': 'Global Financial Crisis', 'severity': 1.0},
            
            # 2010s
            {'start': '2010-05-06', 'end': '2010-05-06', 'type': 'MARKET_CRASH', 'name': 'Flash Crash', 'severity': 0.6},
            {'start': '2011-07-01', 'end': '2012-07-01', 'type': 'CURRENCY_CRISIS', 'name': 'European Debt Crisis', 'severity': 0.75},
            {'start': '2015-08-24', 'end': '2015-08-25', 'type': 'MARKET_CRASH', 'name': 'Chinese Black Monday', 'severity': 0.65},
            {'start': '2018-02-05', 'end': '2018-02-08', 'type': 'MARKET_CRASH', 'name': 'Volmageddon', 'severity': 0.6},
            
            # 2020s
            {'start': '2020-02-19', 'end': '2020-03-23', 'type': 'PANDEMIC', 'name': 'COVID-19 Crash', 'severity': 0.95},
            {'start': '2022-01-03', 'end': '2022-10-12', 'type': 'MARKET_CRASH', 'name': '2022 Bear Market', 'severity': 0.75},
            {'start': '2023-03-08', 'end': '2023-03-20', 'type': 'FINANCIAL_CRISIS', 'name': 'Banking Crisis 2023', 'severity': 0.7},
        ]
        
        return events
    
    def label_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label historical data with crisis events"""
        df = df.copy()
        
        # Initialize columns
        df['is_crisis'] = 0
        df['crisis_type'] = 'NORMAL'
        df['crisis_severity'] = 0.0
        df['days_to_crisis'] = 999
        df['days_in_crisis'] = 0
        
        # Label each event
        for event in self.events:
            start_date = pd.to_datetime(event['start'])
            end_date = pd.to_datetime(event['end'])
            
            # Mark crisis period
            crisis_mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            df.loc[crisis_mask, 'is_crisis'] = 1
            df.loc[crisis_mask, 'crisis_type'] = event['type']
            df.loc[crisis_mask, 'crisis_severity'] = event['severity']
            
            # Calculate days in crisis
            for idx in df[crisis_mask].index:
                days_in = (df.loc[idx, 'Date'] - start_date).days
                df.loc[idx, 'days_in_crisis'] = days_in
            
            # Mark pre-crisis period (30 days before)
            pre_crisis_start = start_date - timedelta(days=30)
            pre_crisis_mask = (df['Date'] >= pre_crisis_start) & (df['Date'] < start_date)
            
            for idx in df[pre_crisis_mask].index:
                days_to = (start_date - df.loc[idx, 'Date']).days
                if days_to < df.loc[idx, 'days_to_crisis']:
                    df.loc[idx, 'days_to_crisis'] = days_to
        
        return df


if __name__ == "__main__":
    # Test data collection
    collector = MarketDataCollector()
    data = collector.collect_all_data()
    
    print("Data collection test completed!")
    print(f"VIX: {data['vix']}")
    print(f"Economic Indicators: {data['economic_indicators']}")
    print(f"News Sentiment: {data['news_sentiment']}")
