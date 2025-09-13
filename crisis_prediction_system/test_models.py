"""
Simple test script to verify model loading and prediction
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crisis_predictor import CrisisPredictionSystem
import warnings
warnings.filterwarnings('ignore')

def test_models():
    print("=" * 80)
    print("TESTING CRISIS PREDICTION MODELS")
    print("=" * 80)
    
    # Initialize system
    print("\n1. Initializing system...")
    predictor = CrisisPredictionSystem()
    
    # Load models
    print("\n2. Loading pre-trained models...")
    success = predictor.load_models()
    
    if success:
        print("✅ Models loaded successfully!")
        
        # Test prediction
        print("\n3. Testing prediction capability...")
        
        # Create mock data for testing
        import pandas as pd
        mock_data = {
            'market_data': pd.DataFrame({'dummy': [1]}),  # Non-empty dataframe
            'vix': 18.5,  # Normal VIX
            'economic_indicators': {
                'Yield Curve': 1.2,
                '10-Year Treasury': 3.5,
                'Unemployment Rate': 3.8
            },
            'news_sentiment': {
                'average_sentiment': 0.1,
                'negative_ratio': 0.3,
                'article_count': 50
            },
            'currency_data': {
                'DXY_proxy': 1.02
            },
            'sector_performance': None
        }
        
        # Override the data collector for testing
        predictor.data_collector.collect_all_data = lambda: mock_data
        
        # Make prediction
        print("\n4. Making test prediction...")
        prediction = predictor.predict_live()
        
        if prediction:
            print("\n✅ Prediction successful!")
            print(f"\nResults:")
            print(f"  30-Day Crisis Probability: {prediction['predictions']['30_day_probability']:.1%}")
            print(f"  6-Month Crisis Probability: {prediction['predictions']['6_month_probability']:.1%}")
            print(f"  Risk Level: {prediction['predictions']['risk_level']}")
            print(f"\nMarket Indicators:")
            for k, v in prediction['market_indicators'].items():
                print(f"  {k}: {v}")
        else:
            print("❌ Prediction failed!")
    else:
        print("❌ Failed to load models!")
        print("\nPlease ensure you have run create_minimal_models.py first")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_models()
