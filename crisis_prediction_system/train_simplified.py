"""
Simplified training script that uses a smaller dataset sample
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crisis_predictor import CrisisPredictionSystem
from datetime import datetime
import json
import pandas as pd

def main():
    print("=" * 80)
    print("CRISIS PREDICTION SYSTEM - SIMPLIFIED MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    # Initialize the system
    print("\n1. Initializing Crisis Prediction System...")
    predictor = CrisisPredictionSystem()
    
    # Prepare data with sampling
    print("\n2. Preparing historical data with sampling...")
    df = predictor.prepare_historical_data()
    
    if df is None:
        print("Failed to load historical data")
        return
    
    # Sample data to reduce memory usage
    print(f"\nOriginal data size: {len(df):,} records")
    
    # Take a stratified sample to ensure we have crisis and non-crisis examples
    crisis_data = df[df['is_crisis'] == 1]
    normal_data = df[df['is_crisis'] == 0]
    
    # Sample 10% of crisis data and a matching amount of normal data
    n_crisis_samples = min(len(crisis_data), 50000)
    n_normal_samples = min(len(normal_data), 100000)
    
    sampled_crisis = crisis_data.sample(n=n_crisis_samples, random_state=42)
    sampled_normal = normal_data.sample(n=n_normal_samples, random_state=42)
    
    # Combine samples
    df_sampled = pd.concat([sampled_crisis, sampled_normal]).sort_values('Date')
    print(f"Sampled data size: {len(df_sampled):,} records")
    print(f"Crisis examples: {len(sampled_crisis):,}")
    print(f"Normal examples: {len(sampled_normal):,}")
    
    # Train models on sampled data
    print("\n3. Training models on sampled data...")
    try:
        success = predictor.train_models(df_sampled)
        
        if success:
            print("\n✅ Models trained successfully!")
            
            # Test the system with a live prediction
            print("\n4. Testing live prediction capabilities...")
            prediction = predictor.predict_live()
            
            if prediction:
                print("\n✅ Live prediction test successful!")
                print(f"\nCurrent Predictions:")
                print(f"  30-Day Crisis Probability: {prediction['predictions']['30_day_probability']:.1%}")
                print(f"  6-Month Crisis Probability: {prediction['predictions']['6_month_probability']:.1%}")
                print(f"  Risk Level: {prediction['predictions']['risk_level']}")
                
                # Save test results
                with open('models/test_results.json', 'w') as f:
                    json.dump({
                        'training_completed': True,
                        'training_samples': len(df_sampled),
                        'test_prediction': {
                            '30_day_probability': prediction['predictions']['30_day_probability'],
                            '6_month_probability': prediction['predictions']['6_month_probability'],
                            'risk_level': prediction['predictions']['risk_level']
                        },
                        'timestamp': str(datetime.now())
                    }, f, indent=2)
            
            print("\n" + "=" * 80)
            print("✅ TRAINING AND TESTING COMPLETE!")
            print("=" * 80)
            print("\nThe following files have been created:")
            print("  - models/xgboost_model.pkl")
            print("  - models/lstm_model.pt (if trained)")
            print("  - models/scalers.pkl")
            print("  - models/feature_columns.json")
            print("  - models/test_results.json")
            print("\n🚀 You can now run 'streamlit run app.py' to launch the web app!")
            print("   The models will be automatically loaded.")
            
        else:
            print("\n❌ Model training failed!")
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nTrying with even smaller sample...")
        
        # Try with an even smaller sample
        df_tiny = df_sampled.sample(n=min(10000, len(df_sampled)), random_state=42)
        success = predictor.train_models(df_tiny)
        
        if success:
            print("\n✅ Models trained successfully with reduced sample!")
            # Save minimal training info
            with open('models/test_results.json', 'w') as f:
                json.dump({
                    'training_completed': True,
                    'training_samples': len(df_tiny),
                    'reduced_sample': True,
                    'timestamp': str(datetime.now())
                }, f, indent=2)
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
