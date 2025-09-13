import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    """
    Comprehensive correlation analysis for the entire stock dataset
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.parquet_file = self.dataset_path / "all_stock_data.parquet"
        self.df = None
        
    def load_full_dataset(self):
        """Load the entire dataset"""
        print("Loading entire dataset...")
        print("This may take a few minutes for 34+ million records...")
        
        try:
            self.df = pd.read_parquet(self.parquet_file, engine='pyarrow')
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            print(f"✓ Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
            print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            print(f"Number of tickers: {self.df['Ticker'].nunique()}")
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def basic_feature_correlations(self):
        """Create correlation heatmap for basic OHLCV features"""
        print("\n=== BASIC FEATURE CORRELATIONS ===")
        
        # Select numerical columns
        numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        available_cols = [col for col in numerical_cols if col in self.df.columns]
        
        # Calculate correlation matrix
        corr_matrix = self.df[available_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        plt.title('Basic Features Correlation Matrix\n(All Stocks, All Time Periods)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_basic_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Basic feature correlation heatmap saved as 'correlation_basic_features.png'")
        return corr_matrix
    
    def calculate_returns(self):
        """Calculate daily returns for correlation analysis"""
        print("Calculating daily returns...")
        
        # Sort by ticker and date
        self.df = self.df.sort_values(['Ticker', 'Date'])
        
        # Calculate daily returns
        self.df['Daily_Return'] = self.df.groupby('Ticker')['Close'].pct_change()
        
        # Calculate log returns (more stable for correlation analysis)
        self.df['Log_Return'] = np.log(self.df['Close'] / self.df.groupby('Ticker')['Close'].shift(1))
        
        # Remove infinite and NaN values
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        print("✓ Returns calculated")
    
    def stock_returns_correlation(self, top_n_stocks=50):
        """Create correlation matrix for top N stocks' returns"""
        print(f"\n=== TOP {top_n_stocks} STOCKS RETURNS CORRELATION ===")
        
        # Get top N stocks by data availability (most trading days)
        stock_counts = self.df['Ticker'].value_counts()
        top_stocks = stock_counts.head(top_n_stocks).index.tolist()
        
        print(f"Analyzing correlations for: {top_stocks[:10]}... (and {len(top_stocks)-10} more)")
        
        # Create pivot table of returns
        returns_pivot = self.df[self.df['Ticker'].isin(top_stocks)].pivot_table(
            index='Date', 
            columns='Ticker', 
            values='Daily_Return'
        )
        
        # Calculate correlation matrix
        returns_corr = returns_pivot.corr()
        
        # Create heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(returns_corr, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   cbar_kws={"shrink": .8})
        plt.title(f'Daily Returns Correlation Matrix\nTop {top_n_stocks} Stocks by Data Availability', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'correlation_top_{top_n_stocks}_stocks.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Top {top_n_stocks} stocks correlation heatmap saved")
        
        # Summary statistics
        print(f"\nCorrelation Summary:")
        print(f"Average correlation: {returns_corr.values[np.triu_indices_from(returns_corr.values, k=1)].mean():.3f}")
        print(f"Max correlation: {returns_corr.values[np.triu_indices_from(returns_corr.values, k=1)].max():.3f}")
        print(f"Min correlation: {returns_corr.values[np.triu_indices_from(returns_corr.values, k=1)].min():.3f}")
        
        return returns_corr
    
    def sector_analysis(self):
        """Analyze correlations by identifying potential sectors"""
        print("\n=== SECTOR-BASED CORRELATION ANALYSIS ===")
        
        # Define some major stocks by sector (simplified classification)
        sector_stocks = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'],
            'Consumer': ['AMZN', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'BKR']
        }
        
        sector_correlations = {}
        
        for sector, stocks in sector_stocks.items():
            # Filter stocks that exist in our dataset
            available_stocks = [stock for stock in stocks if stock in self.df['Ticker'].unique()]
            
            if len(available_stocks) < 3:  # Need at least 3 stocks for meaningful correlation
                continue
                
            print(f"Analyzing {sector} sector: {available_stocks}")
            
            # Create returns pivot for this sector
            sector_data = self.df[self.df['Ticker'].isin(available_stocks)]
            sector_pivot = sector_data.pivot_table(
                index='Date', 
                columns='Ticker', 
                values='Daily_Return'
            )
            
            # Calculate correlation
            sector_corr = sector_pivot.corr()
            sector_correlations[sector] = sector_corr
            
            # Create sector-specific heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(sector_corr, 
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       fmt='.3f',
                       cbar_kws={"shrink": .8})
            plt.title(f'{sector} Sector - Daily Returns Correlation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'correlation_sector_{sector.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print sector summary
            avg_corr = sector_corr.values[np.triu_indices_from(sector_corr.values, k=1)].mean()
            print(f"  Average intra-sector correlation: {avg_corr:.3f}")
        
        print(f"✓ Sector correlation analysis completed for {len(sector_correlations)} sectors")
        return sector_correlations
    
    def time_period_correlations(self):
        """Analyze how correlations change over different time periods"""
        print("\n=== TIME PERIOD CORRELATION ANALYSIS ===")
        
        # Define time periods
        periods = {
            'Pre-2000': ('1962-01-01', '1999-12-31'),
            '2000s': ('2000-01-01', '2009-12-31'),
            '2010s': ('2010-01-01', '2019-12-31'),
            '2020s': ('2020-01-01', '2024-12-31')
        }
        
        # Select a subset of major stocks for this analysis
        major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'XOM', 'GE']
        available_major_stocks = [stock for stock in major_stocks if stock in self.df['Ticker'].unique()]
        
        period_correlations = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, (period_name, (start_date, end_date)) in enumerate(periods.items()):
            print(f"Analyzing {period_name} period...")
            
            # Filter data for this period
            period_mask = (self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)
            period_data = self.df[period_mask & self.df['Ticker'].isin(available_major_stocks)]
            
            if len(period_data) == 0:
                print(f"  No data available for {period_name}")
                continue
            
            # Create pivot table
            period_pivot = period_data.pivot_table(
                index='Date', 
                columns='Ticker', 
                values='Daily_Return'
            )
            
            # Calculate correlation
            period_corr = period_pivot.corr()
            period_correlations[period_name] = period_corr
            
            # Create subplot
            sns.heatmap(period_corr, 
                       ax=axes[i],
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       cbar_kws={"shrink": .8})
            axes[i].set_title(f'{period_name}\nAvg Correlation: {period_corr.values[np.triu_indices_from(period_corr.values, k=1)].mean():.3f}', 
                             fontweight='bold')
        
        plt.suptitle('Stock Correlations Across Different Time Periods', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_time_periods.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Time period correlation analysis completed")
        return period_correlations
    
    def volatility_correlations(self):
        """Analyze correlations during high vs low volatility periods"""
        print("\n=== VOLATILITY-BASED CORRELATION ANALYSIS ===")
        
        # Calculate rolling volatility (20-day window)
        self.df['Rolling_Volatility'] = self.df.groupby('Ticker')['Daily_Return'].rolling(window=20).std().reset_index(0, drop=True)
        
        # Define high and low volatility periods (top/bottom 25%)
        volatility_threshold_high = self.df['Rolling_Volatility'].quantile(0.75)
        volatility_threshold_low = self.df['Rolling_Volatility'].quantile(0.25)
        
        # Select major stocks for analysis
        major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'GE']
        available_stocks = [stock for stock in major_stocks if stock in self.df['Ticker'].unique()]
        
        # High volatility periods
        high_vol_data = self.df[
            (self.df['Rolling_Volatility'] >= volatility_threshold_high) & 
            (self.df['Ticker'].isin(available_stocks))
        ]
        
        # Low volatility periods  
        low_vol_data = self.df[
            (self.df['Rolling_Volatility'] <= volatility_threshold_low) & 
            (self.df['Ticker'].isin(available_stocks))
        ]
        
        # Create correlation matrices
        high_vol_pivot = high_vol_data.pivot_table(index='Date', columns='Ticker', values='Daily_Return')
        low_vol_pivot = low_vol_data.pivot_table(index='Date', columns='Ticker', values='Daily_Return')
        
        high_vol_corr = high_vol_pivot.corr()
        low_vol_corr = low_vol_pivot.corr()
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.heatmap(high_vol_corr, ax=ax1, cmap='RdBu_r', center=0, square=True, 
                   cbar_kws={"shrink": .8})
        ax1.set_title(f'High Volatility Periods\nAvg Correlation: {high_vol_corr.values[np.triu_indices_from(high_vol_corr.values, k=1)].mean():.3f}', 
                     fontweight='bold')
        
        sns.heatmap(low_vol_corr, ax=ax2, cmap='RdBu_r', center=0, square=True, 
                   cbar_kws={"shrink": .8})
        ax2.set_title(f'Low Volatility Periods\nAvg Correlation: {low_vol_corr.values[np.triu_indices_from(low_vol_corr.values, k=1)].mean():.3f}', 
                     fontweight='bold')
        
        plt.suptitle('Stock Correlations: High vs Low Volatility Periods', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_volatility_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Volatility-based correlation analysis completed")
        return high_vol_corr, low_vol_corr
    
    def run_comprehensive_analysis(self):
        """Run all correlation analyses"""
        print("Starting Comprehensive Correlation Analysis")
        print("=" * 60)
        
        # Load dataset
        if not self.load_full_dataset():
            return None
        
        # Calculate returns
        self.calculate_returns()
        
        # Run all analyses
        results = {}
        
        print("\n" + "="*60)
        results['basic_features'] = self.basic_feature_correlations()
        
        print("\n" + "="*60)
        results['stock_returns'] = self.stock_returns_correlation(top_n_stocks=30)
        
        print("\n" + "="*60)
        results['sectors'] = self.sector_analysis()
        
        print("\n" + "="*60)
        results['time_periods'] = self.time_period_correlations()
        
        print("\n" + "="*60)
        high_vol, low_vol = self.volatility_correlations()
        results['volatility'] = {'high_vol': high_vol, 'low_vol': low_vol}
        
        print("\n" + "="*60)
        print("COMPREHENSIVE CORRELATION ANALYSIS COMPLETE!")
        print("Generated correlation heatmaps:")
        print("- correlation_basic_features.png")
        print("- correlation_top_30_stocks.png")
        print("- correlation_sector_*.png (multiple files)")
        print("- correlation_time_periods.png")
        print("- correlation_volatility_comparison.png")
        print("="*60)
        
        return results

if __name__ == "__main__":
    # Path to dataset
    dataset_path = "/Users/abhinavagarwal/.cache/kagglehub/datasets/jakewright/9000-tickers-of-stock-market-data-full-history/versions/2"
    
    # Create analyzer and run comprehensive analysis
    analyzer = CorrelationAnalyzer(dataset_path)
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print("\nAnalysis completed successfully!")
        print("Check the generated PNG files for detailed correlation heatmaps.")
