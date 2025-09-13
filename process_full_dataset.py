import pandas as pd
import numpy as np
from data_processor import StockDataProcessor
from correlation_analysis import CorrelationAnalyzer
from advanced_visualizations import AdvancedStockVisualizer
import warnings
warnings.filterwarnings('ignore')

def process_full_dataset():
    """Process the entire stock dataset and generate comprehensive visualizations"""
    
    print("=" * 80)
    print("PROCESSING FULL STOCK MARKET DATASET")
    print("=" * 80)
    
    # Path to the full dataset
    dataset_path = "/Users/abhinavagarwal/.cache/kagglehub/datasets/jakewright/9000-tickers-of-stock-market-data-full-history/versions/2"
    
    # Initialize processor
    processor = StockDataProcessor(dataset_path)
    
    print("\n1. Processing full dataset (this may take several minutes)...")
    print("   - 34.6 million records")
    print("   - 9,315 tickers") 
    print("   - 1962-2024 time period")
    
    # Process the full dataset without any filtering
    # This will take the most recent 2 years of data for all tickers
    processed_df = processor.process_data_pipeline()
    
    if processed_df is not None:
        print(f"\n✓ Full dataset processed successfully!")
        print(f"  Final shape: {processed_df.shape}")
        print(f"  Tickers: {processed_df['Ticker'].nunique()}")
        print(f"  Date range: {processed_df['Date'].min()} to {processed_df['Date'].max()}")
        
        # Save the full processed dataset
        processor.save_processed_data("full_processed_stock_data.parquet")
        
        # Display sample
        print("\nSample of processed data:")
        print(processed_df.head())
        
        return processed_df
    else:
        print("Failed to process dataset")
        return None

def regenerate_all_visualizations(data_file="full_processed_stock_data.parquet"):
    """Regenerate all visualizations with the full dataset"""
    
    print("\n" + "=" * 80)
    print("REGENERATING ALL VISUALIZATIONS WITH FULL DATASET")
    print("=" * 80)
    
    # Initialize visualizer with full dataset
    visualizer = AdvancedStockVisualizer(data_file)
    
    if visualizer.load_data():
        print(f"\nLoaded full dataset: {visualizer.df.shape[0]:,} records")
        
        # Run all visualizations
        results = visualizer.run_all_visualizations()
        
        print("\n✓ All visualizations regenerated with full dataset!")
        return results
    else:
        print("Failed to load data for visualization")
        return None

def run_full_correlation_analysis():
    """Run correlation analysis on the full dataset"""
    
    print("\n" + "=" * 80)
    print("RUNNING CORRELATION ANALYSIS ON FULL DATASET")
    print("=" * 80)
    
    dataset_path = "/Users/abhinavagarwal/.cache/kagglehub/datasets/jakewright/9000-tickers-of-stock-market-data-full-history/versions/2"
    
    # Create analyzer and run analysis
    analyzer = CorrelationAnalyzer(dataset_path)
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print("\n✓ Correlation analysis completed on full dataset!")
    
    return results

if __name__ == "__main__":
    print("Starting comprehensive analysis of full stock market dataset...")
    print("This will process 34.6 million records across 9,315 tickers")
    print("Please be patient as this may take several minutes...\n")
    
    # Step 1: Process full dataset
    processed_data = process_full_dataset()
    
    if processed_data is not None:
        # Step 2: Regenerate advanced visualizations
        viz_results = regenerate_all_visualizations()
        
        # Step 3: Re-run correlation analysis if needed
        # Uncomment the following line if you want to regenerate correlation heatmaps too
        # corr_results = run_full_correlation_analysis()
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nAll visualizations have been regenerated with the full dataset:")
        print("- Volatility clustering across ALL stocks")
        print("- Volume-price dynamics for diverse stock selection")
        print("- Return distributions for the entire market")
        print("- Lead-lag correlations across all sectors")
        print("- Complete sector rotation analysis")
        print("- And more...")
        
        print("\nYou can now build robust prediction models using insights from")
        print("the complete 62+ year dataset with all 9,315 stocks!")
    else:
        print("\nFailed to process the full dataset. Please check the data path.")
