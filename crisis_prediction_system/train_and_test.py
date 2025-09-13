"""
Script to train all models and test the system
This will create pre-trained models that can be loaded by the Streamlit app
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

def main():
    print("=" * 80)
    print("CRISIS PREDICTION SYSTEM - MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    # Initialize the system
    print("\n1. Initializing Crisis Prediction System...")
    predictor = CrisisPredictionSystem()
    
    # Train all models
    print("\n2. Training models on historical data...")
    print("   This will take approximately 5-10 minutes...")
    
    try:
        success = predictor.train_models()
        
        if success:
            print("\n✅ Models trained successfully!")
            
            # Test the system with a live prediction
            print("\n3. Testing live prediction capabilities...")
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
                        'test_prediction': {
                            '30_day_probability': prediction['predictions']['30_day_probability'],
                            '6_month_probability': prediction['predictions']['6_month_probability'],
                            'risk_level': prediction['predictions']['risk_level']
                        },
                        'timestamp': str(datetime.now())
                    }, f, indent=2)
                
                # Run a quick backtest
                print("\n4. Running backtest on recent events...")
                backtest_results = predictor.backtest(
                    start_date='2019-01-01',
                    end_date='2023-12-31'
                )
                
                if backtest_results:
                    print(f"\nBacktest Results (2019-2023):")
                    print(f"  Total Events: {backtest_results['total_events']}")
                    print(f"  Correctly Predicted: {backtest_results['correctly_predicted']}")
                    print(f"  Accuracy: {backtest_results['accuracy']:.1%}")
                    
                    # Save backtest results
                    with open('models/backtest_results.json', 'w') as f:
                        json.dump(backtest_results, f, indent=2)
            
            print("\n" + "=" * 80)
            print("✅ TRAINING AND TESTING COMPLETE!")
            print("=" * 80)
            print("\nThe following files have been created:")
            print("  - models/xgboost_model.pkl")
            print("  - models/lstm_model.pt (if sufficient data)")
            print("  - models/scalers.pkl")
            print("  - models/feature_columns.json")
            print("  - models/test_results.json")
            print("  - models/backtest_results.json")
            print("\n🚀 You can now run 'streamlit run app.py' to launch the web app!")
            print("   The models will be automatically loaded.")
            
        else:
            print("\n❌ Model training failed!")
            print("Please check that the historical data file exists:")
            print("  ../full_processed_stock_data.parquet")
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all required packages are installed: pip install -r requirements.txt")
        print("2. Check that the historical data file exists")
        print("3. Verify you have sufficient memory (at least 4GB RAM)")
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
