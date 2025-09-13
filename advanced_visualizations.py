import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedStockVisualizer:
    """
    Advanced visualizations for stock market analysis to uncover patterns
    for robust prediction model development.
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        
    def load_data(self, filepath=None):
        """Load stock data from parquet file"""
        if filepath:
            self.df = pd.read_parquet(filepath)
        elif self.data_path:
            self.df = pd.read_parquet(self.data_path)
        else:
            # Try to load from default location
            try:
                self.df = pd.read_parquet('processed_stock_features.parquet')
            except:
                print("Please provide a data file path")
                return False
        
        # Ensure Date column is datetime
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        print(f"Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        return True
    
    def volatility_clustering_heatmap(self, top_n_stocks=30, window=20):
        """
        Create volatility clustering heatmap to identify regime patterns
        """
        print("Generating volatility clustering heatmap...")
        
        # Calculate returns if not present
        if 'Daily_Return' not in self.df.columns:
            self.df['Daily_Return'] = self.df.groupby('Ticker')['Close'].pct_change()
        
        # Get top N stocks by data availability
        stock_counts = self.df['Ticker'].value_counts()
        top_stocks = stock_counts.head(top_n_stocks).index.tolist()
        
        # Filter data for top stocks
        df_filtered = self.df[self.df['Ticker'].isin(top_stocks)].copy()
        
        # Calculate rolling volatility
        df_filtered['Rolling_Vol'] = df_filtered.groupby('Ticker')['Daily_Return'].rolling(
            window=window, min_periods=10
        ).std().reset_index(0, drop=True)
        
        # Create pivot table for heatmap
        volatility_pivot = df_filtered.pivot_table(
            index='Date', 
            columns='Ticker', 
            values='Rolling_Vol'
        )
        
        # Resample to weekly for better visualization
        volatility_weekly = volatility_pivot.resample('W').mean()
        
        # Create the heatmap
        plt.figure(figsize=(20, 12))
        
        # Use a mask for missing values
        mask = volatility_weekly.isnull()
        
        sns.heatmap(
            volatility_weekly.T, 
            cmap='YlOrRd',
            cbar_kws={'label': 'Rolling Volatility (20-day)'},
            mask=mask.T,
            yticklabels=True,
            xticklabels=False
        )
        
        plt.title(f'Volatility Clustering Heatmap - Top {top_n_stocks} Stocks\n(Weekly Averages)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Stock Ticker', fontsize=12)
        
        # Add vertical lines for major events
        ax = plt.gca()
        
        # Major market events
        events = {
            '2008-09-15': 'Lehman Collapse',
            '2020-03-01': 'COVID-19 Crash',
            '2022-01-01': 'Fed Tightening'
        }
        
        for date, label in events.items():
            try:
                event_date = pd.to_datetime(date)
                if event_date in volatility_weekly.index:
                    x_pos = volatility_weekly.index.get_loc(event_date)
                    ax.axvline(x=x_pos, color='blue', linestyle='--', alpha=0.7)
                    ax.text(x_pos, -1, label, rotation=90, ha='right', va='bottom')
            except:
                pass
        
        plt.tight_layout()
        plt.savefig('volatility_clustering_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Volatility clustering heatmap saved")
        
        # Calculate volatility regime statistics
        vol_stats = {
            'mean_volatility': volatility_weekly.mean().mean(),
            'volatility_of_volatility': volatility_weekly.std().mean(),
            'max_volatility': volatility_weekly.max().max(),
            'high_vol_periods': (volatility_weekly > volatility_weekly.mean().mean() * 2).sum().sum()
        }
        
        return volatility_weekly, vol_stats
    
    def volume_price_dynamics(self, ticker_list=None):
        """
        Create volume-price scatter plots to leverage the independence found in correlation analysis
        """
        print("Generating volume-price dynamics plots...")
        
        if ticker_list is None:
            # Select diverse tickers from different sectors
            ticker_list = ['AAPL', 'JPM', 'XOM', 'AMZN', 'JNJ']
            ticker_list = [t for t in ticker_list if t in self.df['Ticker'].unique()][:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, ticker in enumerate(ticker_list):
            if idx >= 4:
                break
                
            ticker_data = self.df[self.df['Ticker'] == ticker].copy()
            
            # Calculate price change and volume ratio
            ticker_data['Price_Change'] = ticker_data['Close'].pct_change()
            ticker_data['Volume_Ratio'] = ticker_data['Volume'] / ticker_data['Volume'].rolling(20).mean()
            
            # Remove outliers for better visualization
            ticker_data = ticker_data[
                (ticker_data['Price_Change'].abs() < ticker_data['Price_Change'].abs().quantile(0.99)) &
                (ticker_data['Volume_Ratio'] < ticker_data['Volume_Ratio'].quantile(0.99))
            ]
            
            ax = axes[idx]
            
            # Create scatter plot with color based on return direction
            colors = ['red' if x < 0 else 'green' for x in ticker_data['Price_Change']]
            
            scatter = ax.scatter(
                ticker_data['Volume_Ratio'], 
                ticker_data['Price_Change'] * 100,
                c=colors, 
                alpha=0.5,
                s=30
            )
            
            # Add trend line
            z = np.polyfit(ticker_data['Volume_Ratio'].dropna(), 
                          ticker_data['Price_Change'].dropna() * 100, 1)
            p = np.poly1d(z)
            ax.plot(ticker_data['Volume_Ratio'].sort_values(), 
                   p(ticker_data['Volume_Ratio'].sort_values()), 
                   "b--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = ticker_data[['Volume_Ratio', 'Price_Change']].corr().iloc[0, 1]
            
            ax.set_title(f'{ticker} - Volume vs Price Change\nCorrelation: {corr:.3f}', 
                        fontweight='bold')
            ax.set_xlabel('Volume Ratio (vs 20-day avg)')
            ax.set_ylabel('Price Change (%)')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=1, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Volume-Price Dynamics Analysis\n(Leveraging Volume Independence for Prediction)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('volume_price_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Volume-price dynamics plots saved")
    
    def return_distribution_analysis(self, ticker_list=None):
        """
        Analyze return distributions to understand tail risks and non-normality
        """
        print("Generating return distribution analysis...")
        
        if ticker_list is None:
            # Get diverse set of stocks
            ticker_list = ['SPY', 'AAPL', 'JPM', 'XOM', 'TSLA', 'GE']
            ticker_list = [t for t in ticker_list if t in self.df['Ticker'].unique()][:6]
        
        fig = plt.figure(figsize=(20, 12))
        
        # Main distribution plots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        for idx, ticker in enumerate(ticker_list):
            if idx >= 6:
                break
            
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Get returns
            ticker_data = self.df[self.df['Ticker'] == ticker].copy()
            returns = ticker_data['Daily_Return'].dropna()
            
            # Remove extreme outliers for visualization
            returns_clean = returns[
                (returns > returns.quantile(0.001)) & 
                (returns < returns.quantile(0.999))
            ]
            
            # Plot histogram
            n, bins, patches = ax.hist(returns_clean, bins=100, density=True, 
                                      alpha=0.7, color='skyblue', edgecolor='black')
            
            # Fit normal distribution
            mu, sigma = stats.norm.fit(returns_clean)
            x = np.linspace(returns_clean.min(), returns_clean.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                   label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
            
            # Calculate statistics
            skew = stats.skew(returns_clean)
            kurt = stats.kurtosis(returns_clean)
            jarque_bera = stats.jarque_bera(returns_clean)
            
            # Add statistics to plot
            textstr = f'Skew: {skew:.3f}\nKurtosis: {kurt:.3f}\nJB p-val: {jarque_bera[1]:.3f}'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f'{ticker} Daily Returns Distribution', fontweight='bold')
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Q-Q plots in the bottom row
        ax_qq = fig.add_subplot(gs[2, :])
        
        # Combined Q-Q plot for all stocks
        for ticker in ticker_list[:3]:  # Limit to 3 for clarity
            ticker_data = self.df[self.df['Ticker'] == ticker]
            returns = ticker_data['Daily_Return'].dropna()
            
            # Calculate theoretical quantiles
            sorted_returns = np.sort(returns)
            norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
            
            ax_qq.plot(norm_quantiles, sorted_returns, 'o', markersize=3, 
                      alpha=0.6, label=ticker)
        
        # Add reference line
        ax_qq.plot([-4, 4], [-4, 4], 'r--', linewidth=2, label='Normal')
        ax_qq.set_xlabel('Theoretical Quantiles (Normal)')
        ax_qq.set_ylabel('Sample Quantiles')
        ax_qq.set_title('Q-Q Plot: Testing Normality of Returns', fontweight='bold')
        ax_qq.legend()
        ax_qq.grid(True, alpha=0.3)
        ax_qq.set_xlim(-4, 4)
        
        plt.suptitle('Return Distribution Analysis - Tail Risk Assessment', 
                    fontsize=16, fontweight='bold')
        plt.savefig('return_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Return distribution analysis saved")
    
    def lead_lag_correlation_analysis(self, ticker_pairs=None, max_lag=10):
        """
        Analyze lead-lag relationships between stocks to find predictive patterns
        """
        print("Generating lead-lag correlation analysis...")
        
        if ticker_pairs is None:
            # Default pairs based on economic relationships
            ticker_pairs = [
                ('XOM', 'CVX'),  # Same sector
                ('AAPL', 'MSFT'),  # Tech competitors
                ('JPM', 'BAC'),  # Finance sector
                ('SPY', 'AAPL'),  # Market vs individual
            ]
        
        # Filter to available pairs
        available_tickers = self.df['Ticker'].unique()
        ticker_pairs = [(t1, t2) for t1, t2 in ticker_pairs 
                       if t1 in available_tickers and t2 in available_tickers]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (ticker1, ticker2) in enumerate(ticker_pairs[:4]):
            ax = axes[idx]
            
            # Get returns for both tickers
            returns1 = self.df[self.df['Ticker'] == ticker1].set_index('Date')['Daily_Return']
            returns2 = self.df[self.df['Ticker'] == ticker2].set_index('Date')['Daily_Return']
            
            # Align the series
            aligned = pd.concat([returns1, returns2], axis=1, join='inner')
            aligned.columns = [ticker1, ticker2]
            
            # Calculate cross-correlations
            correlations = []
            lags = range(-max_lag, max_lag + 1)
            
            for lag in lags:
                if lag < 0:
                    corr = aligned[ticker1].iloc[:lag].corr(aligned[ticker2].iloc[-lag:])
                elif lag > 0:
                    corr = aligned[ticker1].iloc[lag:].corr(aligned[ticker2].iloc[:-lag])
                else:
                    corr = aligned[ticker1].corr(aligned[ticker2])
                correlations.append(corr)
            
            # Plot cross-correlation function
            ax.bar(lags, correlations, color=['red' if x < 0 else 'blue' for x in lags])
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Add significance bands (approximate)
            n = len(aligned)
            significance_level = 2 / np.sqrt(n)
            ax.axhline(y=significance_level, color='green', linestyle='--', alpha=0.5)
            ax.axhline(y=-significance_level, color='green', linestyle='--', alpha=0.5)
            
            # Find peak correlation
            max_corr_idx = np.argmax(np.abs(correlations))
            max_lag = lags[max_corr_idx]
            max_corr = correlations[max_corr_idx]
            
            ax.set_title(f'{ticker1} vs {ticker2}\nPeak: {max_corr:.3f} at lag {max_lag}', 
                        fontweight='bold')
            ax.set_xlabel(f'Lag (days) - Negative: {ticker1} leads, Positive: {ticker2} leads')
            ax.set_ylabel('Correlation')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, 0.5)
        
        plt.suptitle('Lead-Lag Correlation Analysis\n(Finding Predictive Relationships)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('lead_lag_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Lead-lag correlation analysis saved")
    
    def regime_detection_visualization(self, ticker='SPY', lookback=252):
        """
        Visualize market regimes based on volatility and returns
        """
        print("Generating regime detection visualization...")
        
        # Get market data (use SPY as proxy)
        market_data = self.df[self.df['Ticker'] == ticker].copy()
        market_data = market_data.sort_values('Date').set_index('Date')
        
        # Calculate metrics for regime detection
        market_data['Rolling_Return'] = market_data['Close'].pct_change().rolling(20).mean()
        market_data['Rolling_Vol'] = market_data['Daily_Return'].rolling(20).std()
        market_data['Vol_Percentile'] = market_data['Rolling_Vol'].rolling(lookback).rank(pct=True)
        
        # Define regimes
        market_data['Regime'] = 'Normal'
        market_data.loc[market_data['Vol_Percentile'] > 0.8, 'Regime'] = 'High Volatility'
        market_data.loc[market_data['Vol_Percentile'] < 0.2, 'Regime'] = 'Low Volatility'
        market_data.loc[
            (market_data['Rolling_Return'] < -0.001) & 
            (market_data['Vol_Percentile'] > 0.6), 'Regime'
        ] = 'Bear Market'
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Plot 1: Price with regime coloring
        regime_colors = {
            'Normal': 'blue',
            'High Volatility': 'red',
            'Low Volatility': 'green',
            'Bear Market': 'darkred'
        }
        
        for regime, color in regime_colors.items():
            regime_data = market_data[market_data['Regime'] == regime]
            ax1.scatter(regime_data.index, regime_data['Close'], 
                       c=color, label=regime, alpha=0.6, s=2)
        
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title(f'{ticker} Price with Market Regime Classification', fontweight='bold')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling volatility
        ax2.plot(market_data.index, market_data['Rolling_Vol'], color='purple', linewidth=1)
        ax2.axhline(y=market_data['Rolling_Vol'].mean(), color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(market_data.index, 0, market_data['Rolling_Vol'], alpha=0.3, color='purple')
        ax2.set_ylabel('Rolling Volatility (20-day)', fontsize=12)
        ax2.set_title('Market Volatility Evolution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regime duration
        regime_changes = market_data['Regime'] != market_data['Regime'].shift(1)
        regime_blocks = regime_changes.cumsum()
        
        for regime in regime_colors.keys():
            regime_periods = market_data[market_data['Regime'] == regime].groupby(regime_blocks).size()
            if len(regime_periods) > 0:
                ax3.hist(regime_periods, bins=30, alpha=0.7, label=f'{regime}', 
                        color=regime_colors[regime])
        
        ax3.set_xlabel('Duration (days)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Regime Duration Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Market Regime Detection and Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('regime_detection_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Regime detection visualization saved")
        
        # Calculate regime statistics
        regime_stats = market_data.groupby('Regime').agg({
            'Daily_Return': ['mean', 'std', 'count'],
            'Volume': 'mean'
        })
        
        return regime_stats
    
    def sector_rotation_heatmap(self):
        """
        Create sector rotation heatmap to identify leadership changes
        """
        print("Generating sector rotation heatmap...")
        
        # Define sector mappings (simplified)
        sector_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'C'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'Consumer': ['AMZN', 'WMT', 'PG', 'KO', 'HD'],
            'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON']
        }
        
        # Calculate sector returns
        sector_returns = {}
        available_tickers = self.df['Ticker'].unique()
        
        for sector, tickers in sector_mapping.items():
            # Filter to available tickers
            sector_tickers = [t for t in tickers if t in available_tickers]
            if len(sector_tickers) > 0:
                sector_data = self.df[self.df['Ticker'].isin(sector_tickers)]
                
                # Calculate equal-weighted sector return
                sector_daily = sector_data.groupby('Date')['Daily_Return'].mean()
                sector_returns[sector] = sector_daily
        
        # Create DataFrame of sector returns
        sector_df = pd.DataFrame(sector_returns)
        
        # Calculate rolling relative performance (vs market)
        market_return = sector_df.mean(axis=1)
        relative_performance = pd.DataFrame()
        
        for sector in sector_df.columns:
            relative_performance[sector] = (
                sector_df[sector].rolling(20).mean() - 
                market_return.rolling(20).mean()
            )
        
        # Resample to monthly for better visualization
        monthly_performance = relative_performance.resample('M').mean()
        
        # Create heatmap
        plt.figure(figsize=(16, 10))
        
        sns.heatmap(
            monthly_performance.T,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Relative Performance vs Market'},
            yticklabels=True,
            xticklabels=True
        )
        
        plt.title('Sector Rotation Heatmap\n(Monthly Relative Performance vs Market)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sector', fontsize=12)
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('sector_rotation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Sector rotation heatmap saved")
    
    def feature_importance_evolution(self, n_periods=12):
        """
        Visualize how feature importance changes over time (simulated for demonstration)
        """
        print("Generating feature importance evolution plot...")
        
        # Define features we've been analyzing
        features = [
            'Volume_Ratio', 'RSI', 'MACD', 'BB_Position', 
            'Volatility_20', 'Cross_Stock_Corr', 'Sector_Return',
            'Market_Return', 'News_Sentiment'
        ]
        
        # Simulate feature importance over time (in practice, this would come from model training)
        dates = pd.date_range(end=self.df['Date'].max(), periods=n_periods, freq='M')
        
        # Create synthetic but realistic importance patterns
        importance_data = {}
        
        # Volume becomes more important in high volatility
        importance_data['Volume_Ratio'] = 0.15 + 0.1 * np.sin(np.arange(n_periods) * 0.5)
        
        # Technical indicators vary cyclically
        importance_data['RSI'] = 0.1 + 0.05 * np.cos(np.arange(n_periods) * 0.3)
        importance_data['MACD'] = 0.08 + 0.04 * np.sin(np.arange(n_periods) * 0.4)
        importance_data['BB_Position'] = 0.09 + 0.03 * np.cos(np.arange(n_periods) * 0.6)
        
        # Volatility importance increases over time
        importance_data['Volatility_20'] = 0.12 + 0.02 * np.arange(n_periods) / n_periods
        
        # Cross-stock correlation importance
        importance_data['Cross_Stock_Corr'] = 0.18 + 0.08 * np.sin(np.arange(n_periods) * 0.2)
        
        # Sector and market returns
        importance_data['Sector_Return'] = 0.14 + 0.06 * np.cos(np.arange(n_periods) * 0.35)
        importance_data['Market_Return'] = 0.16 + 0.07 * np.sin(np.arange(n_periods) * 0.25)
        
        # News sentiment
        importance_data['News_Sentiment'] = 0.11 + 0.05 * np.random.randn(n_periods) * 0.1
        
        # Create DataFrame
        importance_df = pd.DataFrame(importance_data, index=dates)
        
        # Normalize so each time period sums to 1
        importance_df = importance_df.div(importance_df.sum(axis=1), axis=0)
        
        # Create stacked area plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Stacked area plot
        ax1.stackplot(importance_df.index, importance_df.T, 
                     labels=importance_df.columns, alpha=0.8)
        ax1.set_ylabel('Feature Importance', fontsize=12)
        ax1.set_title('Feature Importance Evolution Over Time', fontweight='bold')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Heatmap of feature importance
        sns.heatmap(importance_df.T, cmap='YlOrRd', ax=ax2, 
                   cbar_kws={'label': 'Importance'})
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Feature', fontsize=12)
        ax2.set_title('Feature Importance Heatmap', fontweight='bold')
        
        plt.suptitle('Dynamic Feature Importance Analysis\n(Key for Adaptive Models)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Feature importance evolution plot saved")
    
    def market_microstructure_analysis(self):
        """
        Analyze market microstructure patterns
        """
        print("Generating market microstructure analysis...")
        
        # Select a few liquid stocks
        liquid_stocks = ['AAPL', 'MSFT', 'JPM', 'XOM']
        liquid_stocks = [s for s in liquid_stocks if s in self.df['Ticker'].unique()][:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, ticker in enumerate(liquid_stocks):
            ax = axes[idx]
            
            ticker_data = self.df[self.df['Ticker'] == ticker].copy()
            
            # Calculate various microstructure metrics
            ticker_data['Spread'] = ticker_data['High'] - ticker_data['Low']
            ticker_data['Spread_Pct'] = ticker_data['Spread'] / ticker_data['Close'] * 100
            ticker_data['Volume_Per_Dollar'] = ticker_data['Volume'] / (ticker_data['Close'] * ticker_data['Volume'])
            ticker_data['Price_Efficiency'] = abs(ticker_data['Close'] - ticker_data['Open']) / ticker_data['Spread']
            
            # Create 2D histogram of spread vs volume
            x = np.log10(ticker_data['Volume'].clip(lower=1))
            y = ticker_data['Spread_Pct']
            
            # Remove outliers
            mask = (y < y.quantile(0.99)) & (x > 0)
            x, y = x[mask], y[mask]
            
            # Create 2D histogram
            h = ax.hist2d(x, y, bins=50, cmap='YlOrRd', cmin=1)
            cb = plt.colorbar(h[3], ax=ax)
            cb.set_label('Frequency')
            
            ax.set_xlabel('Log10(Volume)')
            ax.set_ylabel('Spread (%)')
            ax.set_title(f'{ticker} - Volume vs Spread', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Market Microstructure Analysis\n(Trading Cost and Liquidity Patterns)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('market_microstructure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Market microstructure analysis saved")
    
    def temporal_patterns_analysis(self):
        """
        Analyze temporal patterns: day of week, monthly effects, etc.
        """
        print("Generating temporal patterns analysis...")
        
        # Add temporal features
        df_temporal = self.df.copy()
        df_temporal['DayOfWeek'] = df_temporal['Date'].dt.dayofweek
        df_temporal['Month'] = df_temporal['Date'].dt.month
        df_temporal['DayOfMonth'] = df_temporal['Date'].dt.day
        df_temporal['Quarter'] = df_temporal['Date'].dt.quarter
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Day of week returns
        ax1 = axes[0, 0]
        dow_returns = df_temporal.groupby('DayOfWeek')['Daily_Return'].agg(['mean', 'std'])
        dow_returns.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        x = np.arange(len(dow_returns))
        ax1.bar(x, dow_returns['mean'] * 100, yerr=dow_returns['std'] * 100 / np.sqrt(1000), 
               capsize=5, color='steelblue', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(dow_returns.index)
        ax1.set_ylabel('Average Daily Return (%)')
        ax1.set_title('Day of Week Effect', fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # 2. Monthly returns (January effect, etc.)
        ax2 = axes[0, 1]
        monthly_returns = df_temporal.groupby('Month')['Daily_Return'].agg(['mean', 'std'])
        monthly_returns.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        x = np.arange(len(monthly_returns))
        ax2.bar(x, monthly_returns['mean'] * 100, yerr=monthly_returns['std'] * 100 / np.sqrt(1000),
               capsize=5, color='coral', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(monthly_returns.index, rotation=45)
        ax2.set_ylabel('Average Daily Return (%)')
        ax2.set_title('Monthly Seasonality', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 3. Turn of month effect
        ax3 = axes[1, 0]
        tom_returns = df_temporal.groupby('DayOfMonth')['Daily_Return'].agg(['mean', 'std'])
        
        ax3.plot(tom_returns.index, tom_returns['mean'] * 100, 'o-', color='green', alpha=0.7)
        ax3.fill_between(tom_returns.index, 
                        (tom_returns['mean'] - tom_returns['std']) * 100,
                        (tom_returns['mean'] + tom_returns['std']) * 100,
                        alpha=0.2, color='green')
        ax3.set_xlabel('Day of Month')
        ax3.set_ylabel('Average Daily Return (%)')
        ax3.set_title('Turn of Month Effect', fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Month Start')
        ax3.axvline(x=28, color='red', linestyle='--', alpha=0.5, label='Month End')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Quarterly patterns
        ax4 = axes[1, 1]
        quarterly_returns = df_temporal.groupby('Quarter')['Daily_Return'].agg(['mean', 'std'])
        quarterly_returns.index = ['Q1', 'Q2', 'Q3', 'Q4']
        
        x = np.arange(len(quarterly_returns))
        ax4.bar(x, quarterly_returns['mean'] * 100, yerr=quarterly_returns['std'] * 100 / np.sqrt(1000),
               capsize=5, color='purple', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(quarterly_returns.index)
        ax4.set_ylabel('Average Daily Return (%)')
        ax4.set_title('Quarterly Effects', fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Temporal Patterns Analysis\n(Calendar Anomalies and Seasonal Effects)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('temporal_patterns_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Temporal patterns analysis saved")
    
    def tail_risk_analysis(self):
        """
        Analyze extreme events and tail risks
        """
        print("Generating tail risk analysis...")
        
        # Select major indices/stocks for analysis
        major_tickers = ['SPY', 'AAPL', 'MSFT', 'JPM', 'XOM']
        major_tickers = [t for t in major_tickers if t in self.df['Ticker'].unique()]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Extreme events frequency
        ax1 = axes[0, 0]
        extreme_events = {}
        
        for ticker in major_tickers[:5]:
            ticker_data = self.df[self.df['Ticker'] == ticker]
            returns = ticker_data['Daily_Return'].dropna()
            
            # Count events beyond different thresholds
            thresholds = [1, 2, 3, 4, 5]  # Standard deviations
            counts = []
            for threshold in thresholds:
                extreme_count = (returns.abs() > threshold * returns.std()).sum()
                counts.append(extreme_count / len(returns) * 100)
            
            ax1.plot(thresholds, counts, 'o-', label=ticker, linewidth=2)
        
        # Add theoretical normal distribution line
        normal_probs = [
            (1 - stats.norm.cdf(threshold) + stats.norm.cdf(-threshold)) * 100 
            for threshold in thresholds
        ]
        ax1.plot(thresholds, normal_probs, 'k--', label='Normal Distribution', linewidth=2)
        
        ax1.set_xlabel('Threshold (Standard Deviations)')
        ax1.set_ylabel('Frequency (%)')
        ax1.set_title('Frequency of Extreme Events', fontweight='bold')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Maximum drawdown analysis
        ax2 = axes[0, 1]
        
        for ticker in major_tickers[:3]:
            ticker_data = self.df[self.df['Ticker'] == ticker].sort_values('Date')
            
            # Calculate cumulative returns
            ticker_data['Cumulative_Return'] = (1 + ticker_data['Daily_Return']).cumprod()
            ticker_data['Running_Max'] = ticker_data['Cumulative_Return'].expanding().max()
            ticker_data['Drawdown'] = (ticker_data['Cumulative_Return'] - ticker_data['Running_Max']) / ticker_data['Running_Max']
            
            # Plot drawdown
            ax2.fill_between(ticker_data['Date'], 0, ticker_data['Drawdown'] * 100, 
                           alpha=0.3, label=f'{ticker}')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Historical Drawdowns', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Value at Risk backtesting
        ax3 = axes[1, 0]
        
        # Calculate VaR for different confidence levels
        confidence_levels = [0.95, 0.99]
        colors = ['blue', 'red']
        
        for i, conf_level in enumerate(confidence_levels):
            violations = []
            
            for ticker in major_tickers[:3]:
                ticker_data = self.df[self.df['Ticker'] == ticker]
                returns = ticker_data['Daily_Return'].dropna()
                
                # Calculate rolling VaR
                window = 252  # 1 year
                var_threshold = returns.rolling(window).quantile(1 - conf_level)
                
                # Count violations
                violations_pct = ((returns < var_threshold).sum() / len(returns)) * 100
                violations.append(violations_pct)
            
            x = np.arange(len(major_tickers[:3]))
            ax3.bar(x + i * 0.35, violations, 0.35, 
                   label=f'{conf_level*100:.0f}% VaR', color=colors[i], alpha=0.7)
        
        ax3.set_xticks(x + 0.35/2)
        ax3.set_xticklabels(major_tickers[:3])
        ax3.set_ylabel('Violation Rate (%)')
        ax3.set_title('VaR Backtesting Results', fontweight='bold')
        ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% Expected')
        ax3.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='1% Expected')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Tail dependence
        ax4 = axes[1, 1]
        
        # Calculate tail dependence between pairs
        pairs = [('AAPL', 'MSFT'), ('JPM', 'BAC'), ('XOM', 'CVX')]
        available_pairs = [(t1, t2) for t1, t2 in pairs 
                          if t1 in self.df['Ticker'].unique() and t2 in self.df['Ticker'].unique()]
        
        tail_correlations = []
        normal_correlations = []
        
        for ticker1, ticker2 in available_pairs[:3]:
            # Get aligned returns
            returns1 = self.df[self.df['Ticker'] == ticker1].set_index('Date')['Daily_Return']
            returns2 = self.df[self.df['Ticker'] == ticker2].set_index('Date')['Daily_Return']
            
            aligned = pd.concat([returns1, returns2], axis=1, join='inner').dropna()
            aligned.columns = [ticker1, ticker2]
            
            # Normal correlation
            normal_corr = aligned.corr().iloc[0, 1]
            normal_correlations.append(normal_corr)
            
            # Tail correlation (bottom 10%)
            threshold = aligned.quantile(0.1)
            tail_data = aligned[(aligned[ticker1] < threshold[ticker1]) | 
                              (aligned[ticker2] < threshold[ticker2])]
            tail_corr = tail_data.corr().iloc[0, 1]
            tail_correlations.append(tail_corr)
        
        x = np.arange(len(available_pairs[:3]))
        width = 0.35
        
        ax4.bar(x - width/2, normal_correlations, width, label='Normal Correlation', alpha=0.7)
        ax4.bar(x + width/2, tail_correlations, width, label='Tail Correlation', alpha=0.7)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{t1}-{t2}' for t1, t2 in available_pairs[:3]])
        ax4.set_ylabel('Correlation')
        ax4.set_title('Tail Dependence Analysis', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Tail Risk Analysis\n(Understanding Extreme Events and Dependencies)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('tail_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Tail risk analysis saved")
    
    def run_all_visualizations(self):
        """
        Run all advanced visualizations
        """
        print("Running comprehensive advanced visualization suite...")
        print("=" * 60)
        
        # Priority visualizations for model building
        visualizations = [
            ("Volatility Clustering", self.volatility_clustering_heatmap),
            ("Volume-Price Dynamics", self.volume_price_dynamics),
            ("Return Distributions", self.return_distribution_analysis),
            ("Lead-Lag Correlations", self.lead_lag_correlation_analysis),
            ("Regime Detection", self.regime_detection_visualization),
            ("Sector Rotation", self.sector_rotation_heatmap),
            ("Feature Importance Evolution", self.feature_importance_evolution),
            ("Market Microstructure", self.market_microstructure_analysis),
            ("Temporal Patterns", self.temporal_patterns_analysis),
            ("Tail Risk Analysis", self.tail_risk_analysis)
        ]
        
        results = {}
        
        for name, func in visualizations:
            print(f"\n{name}:")
            print("-" * 40)
            try:
                result = func()
                results[name] = result
                print(f"✓ {name} completed successfully")
            except Exception as e:
                print(f"✗ {name} failed: {str(e)}")
                results[name] = None
        
        print("\n" + "=" * 60)
        print("ADVANCED VISUALIZATION SUITE COMPLETE!")
        print(f"Successfully generated {sum(1 for r in results.values() if r is not None)} out of {len(visualizations)} visualizations")
        print("=" * 60)
        
        return results


if __name__ == "__main__":
    # Initialize visualizer
    visualizer = AdvancedStockVisualizer()
    
    # Load data
    if visualizer.load_data():
        # Run all visualizations
        results = visualizer.run_all_visualizations()
        
        print("\nAll visualizations have been saved as PNG files in the current directory.")
        print("\nThese advanced visualizations provide critical insights for building robust prediction models:")
        print("- Volatility clustering reveals regime patterns")
        print("- Volume-price dynamics show independent predictive signals")
        print("- Lead-lag correlations identify predictive relationships")
        print("- Regime detection helps with model switching")
        print("- Temporal patterns reveal calendar anomalies")
        print("- Tail risk analysis shows extreme event behaviors")
    else:
        print("Failed to load data. Please ensure 'processed_stock_features.parquet' exists.")
