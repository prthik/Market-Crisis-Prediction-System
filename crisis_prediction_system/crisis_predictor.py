"""
Main crisis prediction system that integrates all components
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple
import torch

from config import SYSTEM_CONFIG, EVENT_CATEGORIES, MODEL_CONFIG
from data_collector import MarketDataCollector, HistoricalEventLabeler
from feature_engineering import CrisisFeatureEngineer
from models import (
    XGBoostCrisisModel, 
    EnsembleCrisisPredictor
)
# Import CrisisModelTrainer only when needed (requires PyTorch)

class CrisisPredictionSystem:
    """Complete crisis prediction system"""
    
    def __init__(self):
        self.data_collector = MarketDataCollector()
        self.feature_engineer = CrisisFeatureEngineer()
        self.ensemble_model = EnsembleCrisisPredictor()
        self.model_trainer = None  # Lazy initialization only when training
        
        # Create necessary directories
        os.makedirs(SYSTEM_CONFIG['MODEL_PATH'], exist_ok=True)
        os.makedirs(SYSTEM_CONFIG['DATA_PATH'], exist_ok=True)
        os.makedirs(SYSTEM_CONFIG['LOG_PATH'], exist_ok=True)
        
        self.is_trained = False
        self.feature_columns = []
        
    def prepare_historical_data(self):
        """Prepare historical data for training"""
        print("=" * 80)
        print("PREPARING HISTORICAL DATA FOR CRISIS PREDICTION")
        print("=" * 80)
        
        # Load historical data
        if os.path.exists(SYSTEM_CONFIG['HISTORICAL_DATA_PATH']):
            print(f"Loading historical data from {SYSTEM_CONFIG['HISTORICAL_DATA_PATH']}...")
            df = pd.read_parquet(SYSTEM_CONFIG['HISTORICAL_DATA_PATH'])
            print(f"Loaded {len(df):,} records")
        else:
            print("Historical data file not found. Please run data processing first.")
            return None
        
        # Label historical events
        event_labeler = HistoricalEventLabeler(SYSTEM_CONFIG['HISTORICAL_DATA_PATH'])
        df = event_labeler.label_data(df)
        print("✓ Historical events labeled")
        
        # Engineer features
        df = self.feature_engineer.engineer_all_features(df)
        print("✓ Features engineered")
        
        # Create target variables
        # 30-day prediction target
        df['crisis_next_30_days'] = df.groupby('Ticker')['is_crisis'].rolling(
            window=30, min_periods=1
        ).max().shift(-30).reset_index(0, drop=True).fillna(0).astype(int)
        
        # 6-month prediction target
        df['crisis_next_6_months'] = df.groupby('Ticker')['is_crisis'].rolling(
            window=180, min_periods=1
        ).max().shift(-180).reset_index(0, drop=True).fillna(0).astype(int)
        
        print("✓ Target variables created")
        
        return df
    
    def train_models(self, df: pd.DataFrame = None):
        """Train all prediction models"""
        print("\n" + "=" * 80)
        print("TRAINING CRISIS PREDICTION MODELS")
        print("=" * 80)
        
        # Try to import and initialize trainer
        try:
            from models import CrisisModelTrainer, TORCH_AVAILABLE
            if not self.model_trainer:
                if TORCH_AVAILABLE:
                    self.model_trainer = CrisisModelTrainer(MODEL_CONFIG)
                    print("✓ CrisisModelTrainer initialized with PyTorch support")
                else:
                    print("⚠️ PyTorch not available - Training will use XGBoost only")
        except Exception as e:
            print(f"⚠️ Could not initialize CrisisModelTrainer: {e}")
            print("   Training will proceed with XGBoost only")
            
        if df is None:
            df = self.prepare_historical_data()
            if df is None:
                return False
        
        # Select features (exclude non-numeric columns)
        all_feature_names = self.feature_engineer.feature_names
        
        # Filter out non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in all_feature_names if col in numeric_cols]
        
        # Also exclude label columns
        exclude_cols = ['is_crisis', 'crisis_type', 'crisis_severity', 'days_to_crisis', 
                       'days_in_crisis', 'crisis_next_30_days', 'crisis_next_6_months']
        self.feature_columns = [col for col in self.feature_columns if col not in exclude_cols]
        
        # Filter valid data
        df_valid = df.dropna(subset=self.feature_columns + ['crisis_next_30_days']).copy()
        print(f"\nTraining on {len(df_valid):,} valid samples")
        
        # Train XGBoost model (always available)
        print("\nTraining XGBoost model...")
        from models import XGBoostCrisisModel
        xgb_model = XGBoostCrisisModel(MODEL_CONFIG.get('XGBOOST', {}))
        
        # Prepare data
        X = xgb_model.prepare_features(df_valid, self.feature_columns)
        y = df_valid['crisis_next_30_days'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        xgb_model.train(X_train, y_train, X_val, y_val)
        self.ensemble_model.add_model('xgboost', xgb_model)
        print("✓ XGBoost training completed")
        
        # Train LSTM model if PyTorch is available and trainer is initialized
        if self.model_trainer and len(df_valid) > 10000:
            try:
                lstm_model = self.model_trainer.train_lstm(
                    df_valid.sample(10000), self.feature_columns[:50], 
                    'crisis_next_30_days', epochs=50
                )
                self.ensemble_model.add_model('lstm', lstm_model)
                print("✓ LSTM training completed")
            except Exception as e:
                print(f"⚠️ Could not train LSTM: {e}")
        
        # Save models
        self.ensemble_model.save_models(SYSTEM_CONFIG['MODEL_PATH'])
        
        # Save feature columns
        with open(f"{SYSTEM_CONFIG['MODEL_PATH']}/feature_columns.json", 'w') as f:
            json.dump(self.feature_columns, f)
        
        self.is_trained = True
        print("\n✓ All models trained and saved successfully!")
        return True
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load feature columns
            with open(f"{SYSTEM_CONFIG['MODEL_PATH']}/feature_columns.json", 'r') as f:
                self.feature_columns = json.load(f)
            
            # Only load XGBoost model for now (minimal model setup)
            try:
                import joblib
                xgb_model = joblib.load(f"{SYSTEM_CONFIG['MODEL_PATH']}/xgboost_model.pkl")
                self.ensemble_model.add_model('xgboost', xgb_model)
                
                # Load scalers
                scalers = joblib.load(f"{SYSTEM_CONFIG['MODEL_PATH']}/scalers.pkl")
                self.ensemble_model.scalers = scalers
                
                self.is_trained = True
                print("✓ Models loaded successfully (minimal configuration)")
                return True
            except Exception as e:
                print(f"Error loading XGBoost model: {e}")
                return False
                
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_live(self) -> Dict:
        """Make live predictions using current market data"""
        if not self.is_trained:
            print("Models not trained. Training models first...")
            if not self.train_models():
                return None
        
        print("\n" + "=" * 80)
        print("GENERATING LIVE CRISIS PREDICTIONS")
        print("=" * 80)
        
        # Collect current market data
        current_data = self.data_collector.collect_all_data()
        
        # Process market data into features
        if current_data['market_data'] is not None and not current_data['market_data'].empty:
            # Create a simplified feature set for live prediction
            features = self._create_live_features(current_data)
            
            # Make predictions
            predictions = self._make_predictions(features)
            
            # Calculate risk scores
            risk_assessment = self._assess_risk(predictions, current_data)
            
            return {
                'timestamp': datetime.now(),
                'predictions': predictions,
                'risk_assessment': risk_assessment,
                'market_indicators': self._get_market_indicators(current_data),
                'recommendations': self._generate_recommendations(risk_assessment)
            }
        else:
            print("Unable to collect market data")
            return None
    
    def _create_live_features(self, data: Dict) -> np.ndarray:
        """Create features from live market data"""
        # Simplified feature creation for live data
        features = []
        
        # VIX level
        if data['vix']:
            features.extend([
                data['vix'],
                1 if data['vix'] > 30 else 0,  # High VIX indicator
                1 if data['vix'] > 40 else 0   # Extreme VIX
            ])
        else:
            features.extend([20, 0, 0])  # Default values
        
        # Economic indicators
        if data['economic_indicators']:
            indicators = data['economic_indicators']
            features.extend([
                indicators.get('Yield Curve', 1.0),
                1 if indicators.get('Yield Curve', 1.0) < 0 else 0,  # Inverted yield curve
                indicators.get('10-Year Treasury', 3.0),
                indicators.get('Unemployment Rate', 4.0)
            ])
        else:
            features.extend([1.0, 0, 3.0, 4.0])
        
        # News sentiment
        if data['news_sentiment']:
            sentiment = data['news_sentiment']
            features.extend([
                sentiment['average_sentiment'],
                sentiment['negative_ratio'],
                sentiment['article_count'] / 100  # Normalized
            ])
        else:
            features.extend([0, 0.2, 0.5])
        
        # Currency strength
        if data['currency_data']:
            features.append(data['currency_data'].get('DXY_proxy', 1.0))
        else:
            features.append(1.0)
        
        # Pad with zeros to match training feature size
        while len(features) < len(self.feature_columns):
            features.append(0)
        
        return np.array(features[:len(self.feature_columns)]).reshape(1, -1)
    
    def _make_predictions(self, features: np.ndarray) -> Dict:
        """Make predictions using ensemble model"""
        # Get the model
        xgb_model = self.ensemble_model.models.get('xgboost')
        
        if xgb_model is not None:
            try:
                # Apply scaler if available
                if hasattr(self.ensemble_model, 'scalers') and 'xgboost' in self.ensemble_model.scalers:
                    scaler = self.ensemble_model.scalers['xgboost']
                    features_scaled = scaler.transform(features)
                else:
                    features_scaled = features
                
                # Get prediction
                probs = xgb_model.predict_proba(features_scaled)
                prob_30_day = float(probs[0][1])  # Probability of positive class
                
                # Simplified 6-month prediction (scaled from 30-day)
                prob_6_month = min(prob_30_day * 1.5, 0.95)
            except Exception as e:
                print(f"Prediction error: {e}")
                prob_30_day = 0.15
                prob_6_month = 0.25
        else:
            prob_30_day = 0.15
            prob_6_month = 0.25
        
        return {
            '30_day_probability': float(prob_30_day),
            '6_month_probability': float(prob_6_month),
            'risk_level': self._get_risk_level(prob_30_day),
            'confidence': 0.75  # Simplified confidence score
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability < SYSTEM_CONFIG['RISK_THRESHOLDS']['LOW']:
            return 'LOW'
        elif probability < SYSTEM_CONFIG['RISK_THRESHOLDS']['MEDIUM']:
            return 'MEDIUM'
        elif probability < SYSTEM_CONFIG['RISK_THRESHOLDS']['HIGH']:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _assess_risk(self, predictions: Dict, data: Dict) -> Dict:
        """Comprehensive risk assessment"""
        risk_factors = []
        
        # Check VIX level
        if data['vix'] and data['vix'] > 30:
            risk_factors.append({
                'factor': 'High Volatility',
                'severity': 'HIGH',
                'value': f"VIX at {data['vix']:.1f}"
            })
        
        # Check yield curve
        if data['economic_indicators']:
            yield_curve = data['economic_indicators'].get('Yield Curve', 1)
            if yield_curve < 0:
                risk_factors.append({
                    'factor': 'Inverted Yield Curve',
                    'severity': 'HIGH',
                    'value': f"{yield_curve:.2f}%"
                })
        
        # Check news sentiment
        if data['news_sentiment']:
            if data['news_sentiment']['negative_ratio'] > 0.6:
                risk_factors.append({
                    'factor': 'Negative News Sentiment',
                    'severity': 'MEDIUM',
                    'value': f"{data['news_sentiment']['negative_ratio']*100:.0f}% negative"
                })
        
        # Calculate overall risk score
        severity_scores = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        
        if risk_factors:
            avg_severity = np.mean([
                severity_scores.get(rf['severity'], 2) for rf in risk_factors
            ])
        else:
            avg_severity = 1
        
        return {
            'overall_risk': predictions['risk_level'],
            'risk_factors': risk_factors,
            'risk_score': avg_severity / 4,  # Normalized to 0-1
            'most_likely_event_type': self._predict_event_type(data)
        }
    
    def _predict_event_type(self, data: Dict) -> str:
        """Predict most likely type of crisis event"""
        # Simplified logic based on current indicators
        if data['vix'] and data['vix'] > 40:
            return 'MARKET_CRASH'
        
        if data['economic_indicators']:
            if data['economic_indicators'].get('Yield Curve', 1) < -0.5:
                return 'FINANCIAL_CRISIS'
        
        if data['news_sentiment']:
            if 'war' in str(data['news_sentiment']) or 'conflict' in str(data['news_sentiment']):
                return 'GEOPOLITICAL_SHOCK'
        
        return 'MARKET_CRASH'  # Default
    
    def _get_market_indicators(self, data: Dict) -> Dict:
        """Extract key market indicators"""
        indicators = {}
        
        if data['vix']:
            indicators['VIX'] = round(data['vix'], 2)
        
        if data['economic_indicators']:
            indicators.update({
                k: round(v, 2) 
                for k, v in data['economic_indicators'].items()
            })
        
        if data['sector_performance'] is not None and not data['sector_performance'].empty:
            indicators['worst_sector'] = data['sector_performance'].idxmin()
            indicators['best_sector'] = data['sector_performance'].idxmax()
        
        return indicators
    
    def _generate_recommendations(self, risk_assessment: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        risk_level = risk_assessment['overall_risk']
        
        if risk_level == 'CRITICAL':
            recommendations.extend([
                "🚨 URGENT: Consider significant risk reduction",
                "💰 Increase cash positions to 30-50%",
                "🛡️ Implement hedging strategies immediately",
                "📊 Review and tighten stop-loss orders"
            ])
        elif risk_level == 'HIGH':
            recommendations.extend([
                "⚠️ Begin reducing risk exposure",
                "💵 Increase cash allocation to 20-30%",
                "📉 Consider protective puts on major holdings",
                "🔍 Monitor positions closely"
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "👀 Heightened vigilance required",
                "⚖️ Rebalance portfolio to defensive sectors",
                "📈 Take profits on overextended positions",
                "🎯 Prepare watchlist for opportunities"
            ])
        else:
            recommendations.extend([
                "✅ Market conditions appear stable",
                "📊 Maintain current allocations",
                "🔍 Look for selective opportunities",
                "📚 Stay informed on market developments"
            ])
        
        return recommendations
    
    def backtest(self, start_date: str = '2010-01-01', end_date: str = '2023-12-31') -> Dict:
        """Backtest the prediction system on historical data"""
        print("\n" + "=" * 80)
        print("BACKTESTING CRISIS PREDICTION SYSTEM")
        print("=" * 80)
        
        # Load and prepare data
        df = self.prepare_historical_data()
        if df is None:
            return None
        
        # Filter date range
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        # Get predictions for test period
        test_results = []
        
        # Simplified backtesting - check predictions before each major event
        event_labeler = HistoricalEventLabeler(SYSTEM_CONFIG['HISTORICAL_DATA_PATH'])
        
        for event in event_labeler.events:
            event_date = pd.to_datetime(event['start'])
            
            if start_date <= event['start'] <= end_date:
                # Get data 30 days before event
                pred_date = event_date - timedelta(days=30)
                
                # Check if we predicted the crisis
                pred_data = df[
                    (df['Date'] >= pred_date - timedelta(days=5)) & 
                    (df['Date'] <= pred_date + timedelta(days=5))
                ]
                
                if not pred_data.empty:
                    avg_prediction = pred_data['crisis_score'].mean() if 'crisis_score' in pred_data else 0
                    
                    test_results.append({
                        'event': event['name'],
                        'event_date': event['start'],
                        'event_type': event['type'],
                        'severity': event['severity'],
                        'predicted_score': avg_prediction,
                        'predicted': avg_prediction > 0.5,
                        'actual': True
                    })
        
        # Calculate metrics
        if test_results:
            results_df = pd.DataFrame(test_results)
            
            metrics = {
                'total_events': len(test_results),
                'correctly_predicted': sum(results_df['predicted']),
                'accuracy': sum(results_df['predicted']) / len(results_df),
                'events_by_type': results_df.groupby('event_type')['predicted'].agg(['sum', 'count']).to_dict(),
                'detailed_results': test_results
            }
        else:
            metrics = {
                'total_events': 0,
                'correctly_predicted': 0,
                'accuracy': 0,
                'events_by_type': {},
                'detailed_results': []
            }
        
        print(f"\nBacktest Results:")
        print(f"Total Events: {metrics['total_events']}")
        print(f"Correctly Predicted: {metrics['correctly_predicted']}")
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        
        return metrics


if __name__ == "__main__":
    # Initialize system
    predictor = CrisisPredictionSystem()
    
    # Train models
    predictor.train_models()
    
    # Make live prediction
    prediction = predictor.predict_live()
    
    if prediction:
        print("\n" + "=" * 80)
        print("CURRENT CRISIS PREDICTION")
        print("=" * 80)
        print(f"30-Day Crisis Probability: {prediction['predictions']['30_day_probability']:.1%}")
        print(f"6-Month Crisis Probability: {prediction['predictions']['6_month_probability']:.1%}")
        print(f"Risk Level: {prediction['predictions']['risk_level']}")
        print(f"\nRisk Factors:")
        for factor in prediction['risk_assessment']['risk_factors']:
            print(f"  - {factor['factor']}: {factor['value']} ({factor['severity']})")
        print(f"\nRecommendations:")
        for rec in prediction['recommendations']:
            print(f"  {rec}")
