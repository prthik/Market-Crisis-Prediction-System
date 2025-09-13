import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class NewsIntegrator:
    """
    News and current events integrator for stock price prediction.
    Fetches news data, performs sentiment analysis, and aligns with stock data.
    """
    
    def __init__(self, news_api_key=None):
        self.news_api_key = news_api_key
        self.news_data = None
        self.processed_news = None
        
    def fetch_news_data(self, tickers, start_date, end_date, sources=None):
        """
        Fetch news data for specific tickers and date range
        """
        print("Fetching news data...")
        
        if not self.news_api_key:
            print("Warning: No News API key provided. Using mock data for demonstration.")
            return self._generate_mock_news_data(tickers, start_date, end_date)
        
        all_news = []
        
        for ticker in tickers:
            try:
                # Fetch news for each ticker
                news_data = self._fetch_ticker_news(ticker, start_date, end_date, sources)
                all_news.extend(news_data)
            except Exception as e:
                print(f"Error fetching news for {ticker}: {e}")
                continue
        
        if all_news:
            self.news_data = pd.DataFrame(all_news)
            print(f"Fetched {len(self.news_data)} news articles")
            return self.news_data
        else:
            print("No news data fetched. Using mock data.")
            return self._generate_mock_news_data(tickers, start_date, end_date)
    
    def _fetch_ticker_news(self, ticker, start_date, end_date, sources=None):
        """
        Fetch news for a specific ticker using News API
        """
        base_url = "https://newsapi.org/v2/everything"
        
        # Convert dates to string format
        start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        
        # Search queries for the ticker
        queries = [
            f"{ticker} stock",
            f"{ticker} earnings",
            f"{ticker} financial"
        ]
        
        ticker_news = []
        
        for query in queries:
            params = {
                'q': query,
                'from': start_str,
                'to': end_str,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.news_api_key,
                'pageSize': 50
            }
            
            if sources:
                params['sources'] = ','.join(sources)
            
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data['status'] == 'ok':
                    for article in data['articles']:
                        ticker_news.append({
                            'ticker': ticker,
                            'title': article['title'],
                            'description': article['description'],
                            'content': article['content'],
                            'published_at': article['publishedAt'],
                            'source': article['source']['name'],
                            'url': article['url']
                        })
                
            except Exception as e:
                print(f"Error fetching news for query '{query}': {e}")
                continue
        
        return ticker_news
    
    def _generate_mock_news_data(self, tickers, start_date, end_date):
        """
        Generate mock news data for demonstration purposes
        """
        print("Generating mock news data for demonstration...")
        
        # Sample news templates with different sentiments
        news_templates = [
            # Positive news
            ("{ticker} reports strong quarterly earnings beating expectations", "positive"),
            ("{ticker} announces new product launch with innovative features", "positive"),
            ("{ticker} stock upgraded by analysts citing strong fundamentals", "positive"),
            ("{ticker} expands into new markets with strategic partnership", "positive"),
            ("{ticker} CEO optimistic about future growth prospects", "positive"),
            
            # Negative news
            ("{ticker} faces regulatory scrutiny over business practices", "negative"),
            ("{ticker} reports disappointing quarterly results", "negative"),
            ("{ticker} stock downgraded due to market concerns", "negative"),
            ("{ticker} announces layoffs amid cost-cutting measures", "negative"),
            ("{ticker} faces supply chain disruptions affecting production", "negative"),
            
            # Neutral news
            ("{ticker} schedules quarterly earnings call for next week", "neutral"),
            ("{ticker} announces board meeting to discuss strategic initiatives", "neutral"),
            ("{ticker} files routine regulatory documents with SEC", "neutral"),
            ("{ticker} participates in industry conference", "neutral"),
            ("{ticker} updates corporate governance policies", "neutral")
        ]
        
        mock_news = []
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for ticker in tickers:
            for date in date_range:
                # Generate 0-3 news articles per day per ticker
                num_articles = np.random.poisson(0.5)  # Low frequency for realistic simulation
                
                for _ in range(num_articles):
                    template, sentiment = np.random.choice(news_templates, 1)[0]
                    title = template.format(ticker=ticker)
                    
                    # Add some randomness to publication time
                    pub_time = date + timedelta(
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    )
                    
                    mock_news.append({
                        'ticker': ticker,
                        'title': title,
                        'description': title + " according to market sources.",
                        'content': f"Full article content about {title}...",
                        'published_at': pub_time.isoformat(),
                        'source': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch']),
                        'url': f"https://example.com/news/{ticker.lower()}-{date.strftime('%Y%m%d')}",
                        'mock_sentiment': sentiment
                    })
        
        self.news_data = pd.DataFrame(mock_news)
        print(f"Generated {len(self.news_data)} mock news articles")
        return self.news_data
    
    def analyze_sentiment(self, df_news):
        """
        Perform sentiment analysis on news articles
        """
        print("Analyzing news sentiment...")
        
        df_sentiment = df_news.copy()
        
        # Initialize sentiment columns
        df_sentiment['sentiment_score'] = 0.0
        df_sentiment['sentiment_label'] = 'neutral'
        
        for idx, row in df_sentiment.iterrows():
            try:
                # Combine title and description for sentiment analysis
                text = f"{row['title']} {row['description']}"
                
                if pd.isna(text) or text.strip() == '':
                    continue
                
                # Use TextBlob for sentiment analysis
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                
                # Classify sentiment
                if sentiment_score > 0.1:
                    sentiment_label = 'positive'
                elif sentiment_score < -0.1:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
                
                df_sentiment.loc[idx, 'sentiment_score'] = sentiment_score
                df_sentiment.loc[idx, 'sentiment_label'] = sentiment_label
                
            except Exception as e:
                print(f"Error analyzing sentiment for article {idx}: {e}")
                continue
        
        print(f"Sentiment analysis completed for {len(df_sentiment)} articles")
        return df_sentiment
    
    def aggregate_daily_sentiment(self, df_sentiment):
        """
        Aggregate news sentiment by ticker and date
        """
        print("Aggregating daily sentiment scores...")
        
        # Convert published_at to datetime and extract date
        df_sentiment['published_date'] = pd.to_datetime(df_sentiment['published_at']).dt.date
        
        # Aggregate by ticker and date
        daily_sentiment = df_sentiment.groupby(['ticker', 'published_date']).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            'ticker', 'date', 'avg_sentiment', 'sentiment_volatility', 
            'news_count', 'dominant_sentiment'
        ]
        
        # Fill NaN sentiment volatility with 0
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_volatility'].fillna(0)
        
        # Convert date back to datetime
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        print(f"Aggregated sentiment for {len(daily_sentiment)} ticker-date combinations")
        return daily_sentiment
    
    def classify_events(self, df_news):
        """
        Classify news articles by event type
        """
        print("Classifying news events...")
        
        df_events = df_news.copy()
        df_events['event_type'] = 'general'
        
        # Define event classification keywords
        event_keywords = {
            'earnings': ['earnings', 'quarterly', 'revenue', 'profit', 'eps'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover'],
            'regulatory': ['regulatory', 'sec', 'investigation', 'lawsuit', 'compliance'],
            'product': ['product', 'launch', 'innovation', 'patent', 'technology'],
            'management': ['ceo', 'executive', 'management', 'leadership', 'board'],
            'partnership': ['partnership', 'collaboration', 'alliance', 'joint venture'],
            'financial': ['debt', 'loan', 'financing', 'ipo', 'dividend', 'buyback']
        }
        
        for idx, row in df_events.iterrows():
            text = f"{row['title']} {row['description']}".lower()
            
            for event_type, keywords in event_keywords.items():
                if any(keyword in text for keyword in keywords):
                    df_events.loc[idx, 'event_type'] = event_type
                    break
        
        print("Event classification completed")
        return df_events
    
    def integrate_with_stock_data(self, stock_df, news_df):
        """
        Integrate news data with stock data
        """
        print("Integrating news data with stock data...")
        
        # Ensure date columns are datetime
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Merge stock data with news data
        integrated_df = stock_df.merge(
            news_df,
            left_on=['Date', 'Ticker'],
            right_on=['date', 'ticker'],
            how='left'
        )
        
        # Fill missing news data with defaults
        news_columns = ['avg_sentiment', 'sentiment_volatility', 'news_count', 'dominant_sentiment']
        for col in news_columns:
            if col in integrated_df.columns:
                if col == 'dominant_sentiment':
                    integrated_df[col] = integrated_df[col].fillna('neutral')
                else:
                    integrated_df[col] = integrated_df[col].fillna(0)
        
        # Drop duplicate date columns
        if 'date' in integrated_df.columns:
            integrated_df = integrated_df.drop('date', axis=1)
        if 'ticker' in integrated_df.columns:
            integrated_df = integrated_df.drop('ticker', axis=1)
        
        # Update the news integration columns in stock data
        integrated_df['News_Sentiment'] = integrated_df.get('avg_sentiment', 0)
        integrated_df['News_Volume'] = integrated_df.get('news_count', 0)
        integrated_df['Event_Type'] = integrated_df.get('dominant_sentiment', 'neutral')
        
        print(f"Integration completed. Final shape: {integrated_df.shape}")
        return integrated_df
    
    def process_news_pipeline(self, stock_df, tickers=None, start_date=None, end_date=None):
        """
        Complete news processing pipeline
        """
        print("Starting news processing pipeline...")
        print("=" * 50)
        
        # Extract parameters from stock data if not provided
        if tickers is None:
            tickers = stock_df['Ticker'].unique().tolist()
        
        if start_date is None:
            start_date = stock_df['Date'].min()
        
        if end_date is None:
            end_date = stock_df['Date'].max()
        
        print(f"Processing news for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Step 1: Fetch news data
        news_df = self.fetch_news_data(tickers, start_date, end_date)
        
        if news_df is None or len(news_df) == 0:
            print("No news data available. Returning stock data with default news features.")
            return stock_df
        
        # Step 2: Analyze sentiment
        news_df = self.analyze_sentiment(news_df)
        
        # Step 3: Classify events
        news_df = self.classify_events(news_df)
        
        # Step 4: Aggregate daily sentiment
        daily_sentiment = self.aggregate_daily_sentiment(news_df)
        
        # Step 5: Integrate with stock data
        integrated_df = self.integrate_with_stock_data(stock_df, daily_sentiment)
        
        self.processed_news = integrated_df
        
        print("=" * 50)
        print("News processing pipeline completed!")
        print(f"Final integrated dataset shape: {integrated_df.shape}")
        
        return integrated_df

# Install required dependency
def install_textblob():
    """Install TextBlob if not available"""
    try:
        import textblob
    except ImportError:
        print("Installing TextBlob for sentiment analysis...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'textblob'])
        print("TextBlob installed successfully!")

if __name__ == "__main__":
    # Example usage
    install_textblob()
    
    # Load processed stock data
    try:
        stock_df = pd.read_parquet('processed_stock_features.parquet')
        print(f"Loaded stock data: {stock_df.shape}")
        
        # Initialize news integrator (without API key for demo)
        news_integrator = NewsIntegrator()
        
        # Process news integration
        integrated_df = news_integrator.process_news_pipeline(stock_df)
        
        # Save integrated data
        integrated_df.to_parquet('integrated_stock_news_data.parquet', index=False)
        print("Integrated data saved to 'integrated_stock_news_data.parquet'")
        
        # Display sample
        print("\nSample of integrated data:")
        sample_cols = ['Date', 'Ticker', 'Close', 'Next_Return', 'News_Sentiment', 'News_Volume', 'Event_Type']
        available_cols = [col for col in sample_cols if col in integrated_df.columns]
        print(integrated_df[available_cols].head(10))
        
    except FileNotFoundError:
        print("Processed stock data not found. Please run data_processor.py first.")
