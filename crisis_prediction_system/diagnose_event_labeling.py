"""
Comprehensive diagnostic script to investigate event labeling issues
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SYSTEM_CONFIG
from data_collector import HistoricalEventLabeler

def diagnose_event_labeling():
    """Diagnose the event labeling issue"""
    print("=" * 80)
    print("EVENT LABELING DIAGNOSTICS")
    print("=" * 80)
    
    # Load data
    data_path = SYSTEM_CONFIG['HISTORICAL_DATA_PATH']
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records")
    
    # Check date range
    print("\n" + "=" * 80)
    print("DATA DATE RANGE")
    print("=" * 80)
    print(f"Earliest date: {df['Date'].min()}")
    print(f"Latest date: {df['Date'].max()}")
    print(f"Date range: {(df['Date'].max() - df['Date'].min()).days} days")
    print(f"Years covered: {(df['Date'].max() - df['Date'].min()).days / 365:.1f} years")
    
    # Check unique years
    years = df['Date'].dt.year.unique()
    print(f"\nUnique years in data: {sorted(years)}")
    print(f"Total years: {len(years)}")
    
    # Check data distribution by decade
    print("\n" + "=" * 80)
    print("DATA DISTRIBUTION BY DECADE")
    print("=" * 80)
    df['Decade'] = (df['Date'].dt.year // 10) * 10
    decade_counts = df['Decade'].value_counts().sort_index()
    for decade, count in decade_counts.items():
        print(f"{decade}s: {count:,} records ({count/len(df)*100:.1f}%)")
    
    # Initialize event labeler
    event_labeler = HistoricalEventLabeler(data_path)
    
    # Check which events should be in the data range
    print("\n" + "=" * 80)
    print("EVENTS vs DATA RANGE")
    print("=" * 80)
    
    data_start = df['Date'].min()
    data_end = df['Date'].max()
    
    events_in_range = []
    events_out_range = []
    
    for event in event_labeler.events:
        event_start = pd.to_datetime(event['start'])
        event_end = pd.to_datetime(event['end'])
        
        # Check if event overlaps with data range
        if event_end >= data_start and event_start <= data_end:
            events_in_range.append(event)
        else:
            events_out_range.append(event)
    
    print(f"\nEvents within data range: {len(events_in_range)}")
    print(f"Events outside data range: {len(events_out_range)}")
    
    print("\nEvents that SHOULD be labeled:")
    for event in events_in_range:
        print(f"  - {event['start']} to {event['end']}: {event['name']}")
    
    if events_out_range:
        print("\nEvents OUTSIDE data range:")
        for event in events_out_range:
            print(f"  - {event['start']} to {event['end']}: {event['name']}")
    
    # Label the data
    print("\n" + "=" * 80)
    print("LABELING RESULTS")
    print("=" * 80)
    
    df_labeled = event_labeler.label_data(df)
    
    # Check labeling results
    crisis_records = df_labeled[df_labeled['is_crisis'] == 1]
    print(f"\nTotal crisis records labeled: {len(crisis_records):,}")
    print(f"Crisis ratio: {len(crisis_records)/len(df)*100:.2%}")
    
    # Check crisis distribution by event
    print("\nCrisis records by type:")
    crisis_type_counts = crisis_records['crisis_type'].value_counts()
    for crisis_type, count in crisis_type_counts.items():
        print(f"  {crisis_type}: {count:,} records")
    
    # Check crisis date ranges
    print("\nActual crisis periods found:")
    crisis_dates = crisis_records.groupby('crisis_type')['Date'].agg(['min', 'max'])
    for crisis_type, (min_date, max_date) in crisis_dates.iterrows():
        print(f"  {crisis_type}: {min_date} to {max_date}")
    
    # Check specific events
    print("\n" + "=" * 80)
    print("SPECIFIC EVENT CHECKS")
    print("=" * 80)
    
    # Check 2008 Financial Crisis
    gfc_start = pd.to_datetime('2007-10-09')
    gfc_end = pd.to_datetime('2009-03-09')
    gfc_data = df_labeled[(df_labeled['Date'] >= gfc_start) & (df_labeled['Date'] <= gfc_end)]
    gfc_crisis = gfc_data[gfc_data['is_crisis'] == 1]
    print(f"\n2008 Financial Crisis (should have ~300+ trading days):")
    print(f"  Data in range: {len(gfc_data):,} records")
    print(f"  Labeled as crisis: {len(gfc_crisis):,} records")
    
    # Check COVID crash
    covid_start = pd.to_datetime('2020-02-19')
    covid_end = pd.to_datetime('2020-03-23')
    covid_data = df_labeled[(df_labeled['Date'] >= covid_start) & (df_labeled['Date'] <= covid_end)]
    covid_crisis = covid_data[covid_data['is_crisis'] == 1]
    print(f"\nCOVID-19 Crash:")
    print(f"  Data in range: {len(covid_data):,} records")
    print(f"  Labeled as crisis: {len(covid_crisis):,} records")
    
    # Sample check - show some records around a crisis date
    print("\n" + "=" * 80)
    print("SAMPLE DATA CHECK")
    print("=" * 80)
    
    # Pick a date that should be in crisis
    sample_date = pd.to_datetime('2008-10-15')  # Middle of 2008 crisis
    sample_data = df_labeled[
        (df_labeled['Date'] >= sample_date - pd.Timedelta(days=5)) & 
        (df_labeled['Date'] <= sample_date + pd.Timedelta(days=5))
    ]
    
    if not sample_data.empty:
        print(f"\nSample data around {sample_date} (2008 crisis):")
        print(sample_data[['Date', 'Ticker', 'is_crisis', 'crisis_type']].head(20))
    
    # Check unique tickers
    print("\n" + "=" * 80)
    print("TICKER ANALYSIS")
    print("=" * 80)
    unique_tickers = df['Ticker'].unique()
    print(f"Total unique tickers: {len(unique_tickers):,}")
    
    # Check if some tickers have limited history
    ticker_date_ranges = df.groupby('Ticker')['Date'].agg(['min', 'max', 'count'])
    
    # Find tickers with recent data only
    recent_only = ticker_date_ranges[ticker_date_ranges['min'] > pd.to_datetime('2020-01-01')]
    print(f"\nTickers with data only after 2020: {len(recent_only):,}")
    
    # Find tickers with long history
    long_history = ticker_date_ranges[ticker_date_ranges['min'] < pd.to_datetime('2010-01-01')]
    print(f"Tickers with data before 2010: {len(long_history):,}")
    
    return df_labeled, events_in_range

if __name__ == "__main__":
    df_labeled, events_in_range = diagnose_event_labeling()
