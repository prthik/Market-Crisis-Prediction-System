"""
Machine learning models for crisis prediction
Includes LSTM, Graph Neural Network, XGBoost, and ensemble models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

class LSTMCrisisPredictor(nn.Module):
    """LSTM model for time series crisis prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 dropout: float = 0.2, num_classes: int = 2):
        super(LSTMCrisisPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # For regression (probability output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last hidden state
        out = attn_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        # For probability output
        prob_output = self.sigmoid(out[:, 0])
        
        return out, prob_output


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for modeling contagion effects"""
    
    def __init__(self, num_features: int, hidden_dim: int = 64, 
                 num_layers: int = 3, dropout: float = 0.3):
        super(GraphNeuralNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(self._create_conv_layer(num_features, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.conv_layers.append(self._create_conv_layer(hidden_dim, hidden_dim))
        
        self.conv_layers.append(self._create_conv_layer(hidden_dim, hidden_dim))
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def _create_conv_layer(self, in_features: int, out_features: int):
        """Create a graph convolution layer"""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, x, adj_matrix):
        """Forward pass through GNN"""
        # Apply graph convolutions
        for i, conv in enumerate(self.conv_layers):
            # Graph convolution: aggregate neighbor features
            x = torch.matmul(adj_matrix, x)
            x = conv(x)
        
        # Global pooling
        x = torch.mean(x, dim=0)
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return torch.sigmoid(x)


class XGBoostCrisisModel:
    """XGBoost model for crisis prediction"""
    
    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            'objective': 'binary:logistic',
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42
        }
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Prepare features for XGBoost"""
        X = df[feature_cols].copy()
        
        # Replace infinite values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with 0
        X = X.fillna(0)
        
        # Clip extreme values
        X = X.clip(lower=-1e10, upper=1e10)
        
        return self.scaler.fit_transform(X.values)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train XGBoost model"""
        # Remove early_stopping_rounds from params if it exists
        params = self.params.copy()
        if 'early_stopping_rounds' in params:
            params.pop('early_stopping_rounds')
            
        self.model = xgb.XGBClassifier(**params)
        
        # For newer XGBoost, just use eval_set without early_stopping_rounds
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        return self.model
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict crisis probability"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        return dict(zip(range(len(importance)), importance))


class EnsembleCrisisPredictor:
    """Ensemble model combining LSTM, GNN, and XGBoost"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'lstm': 0.4,
            'gnn': 0.3,
            'xgboost': 0.3
        }
        
        self.models = {
            'lstm': None,
            'gnn': None,
            'xgboost': None
        }
        
        self.scalers = {
            'lstm': StandardScaler(),
            'gnn': StandardScaler(),
            'xgboost': StandardScaler()
        }
        
    def add_model(self, name: str, model):
        """Add a model to the ensemble"""
        if name not in self.models:
            raise ValueError(f"Unknown model name: {name}")
        self.models[name] = model
        
    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Get ensemble prediction"""
        predictions = {}
        
        # Get predictions from each model
        if self.models['lstm'] is not None and 'lstm' in features:
            lstm_pred = self._predict_lstm(features['lstm'])
            predictions['lstm'] = lstm_pred
            
        if self.models['gnn'] is not None and 'gnn' in features:
            gnn_pred = self._predict_gnn(features['gnn'])
            predictions['gnn'] = gnn_pred
            
        if self.models['xgboost'] is not None and 'xgboost' in features:
            xgb_pred = self._predict_xgboost(features['xgboost'])
            predictions['xgboost'] = xgb_pred
        
        # Weighted average
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0)
            ensemble_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
            
        return ensemble_pred
    
    def _predict_lstm(self, features: torch.Tensor) -> np.ndarray:
        """Get LSTM predictions"""
        self.models['lstm'].eval()
        with torch.no_grad():
            _, probs = self.models['lstm'](features)
        return probs.cpu().numpy()
    
    def _predict_gnn(self, features: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        """Get GNN predictions"""
        x, adj = features
        self.models['gnn'].eval()
        with torch.no_grad():
            probs = self.models['gnn'](x, adj)
        return probs.cpu().numpy()
    
    def _predict_xgboost(self, features: np.ndarray) -> np.ndarray:
        """Get XGBoost predictions"""
        return self.models['xgboost'].predict_proba(features)
    
    def save_models(self, path: str):
        """Save all models"""
        # Save PyTorch models
        if self.models['lstm'] is not None:
            torch.save(self.models['lstm'].state_dict(), f"{path}/lstm_model.pt")
            
        if self.models['gnn'] is not None:
            torch.save(self.models['gnn'].state_dict(), f"{path}/gnn_model.pt")
            
        # Save XGBoost model
        if self.models['xgboost'] is not None:
            joblib.dump(self.models['xgboost'], f"{path}/xgboost_model.pkl")
            
        # Save scalers
        joblib.dump(self.scalers, f"{path}/scalers.pkl")
        
    def load_models(self, path: str, input_sizes: Dict):
        """Load all models"""
        # Load LSTM
        if 'lstm' in input_sizes:
            self.models['lstm'] = LSTMCrisisPredictor(input_size=input_sizes['lstm'])
            self.models['lstm'].load_state_dict(torch.load(f"{path}/lstm_model.pt"))
            
        # Load GNN
        if 'gnn' in input_sizes:
            self.models['gnn'] = GraphNeuralNetwork(num_features=input_sizes['gnn'])
            self.models['gnn'].load_state_dict(torch.load(f"{path}/gnn_model.pt"))
            
        # Load XGBoost
        self.models['xgboost'] = joblib.load(f"{path}/xgboost_model.pkl")
        
        # Load scalers
        self.scalers = joblib.load(f"{path}/scalers.pkl")


class CrisisModelTrainer:
    """Trainer class for all crisis prediction models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_lstm_data(self, df: pd.DataFrame, feature_cols: List[str], 
                         target_col: str, sequence_length: int = 60) -> Tuple:
        """Prepare data for LSTM training"""
        # Sort by date and ticker
        df = df.sort_values(['Ticker', 'Date'])
        
        X_sequences = []
        y_sequences = []
        
        # Create sequences for each ticker
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker]
            
            features = ticker_data[feature_cols].fillna(0).values
            targets = ticker_data[target_col].values
            
            # Create sequences
            for i in range(sequence_length, len(features)):
                X_sequences.append(features[i-sequence_length:i])
                y_sequences.append(targets[i])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        return X_train, X_val, y_train, y_val
    
    def prepare_gnn_data(self, df: pd.DataFrame, feature_cols: List[str], 
                        target_col: str) -> Tuple:
        """Prepare data for GNN training"""
        # Create correlation-based adjacency matrix
        returns_pivot = df.pivot_table(
            index='Date', 
            columns='Ticker', 
            values='returns'
        ).fillna(0)
        
        # Calculate correlation matrix
        corr_matrix = returns_pivot.corr().fillna(0).values
        
        # Create adjacency matrix (threshold correlations)
        adj_matrix = (corr_matrix > 0.5).astype(float)
        np.fill_diagonal(adj_matrix, 1)  # Self-connections
        
        # Prepare features
        latest_data = df.groupby('Ticker').last()
        features = latest_data[feature_cols].fillna(0).values
        targets = latest_data[target_col].values
        
        # Convert to tensors
        X = torch.FloatTensor(features).to(self.device)
        adj = torch.FloatTensor(adj_matrix).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        
        return X, adj, y
    
    def train_lstm(self, df: pd.DataFrame, feature_cols: List[str], 
                   target_col: str, epochs: int = 100) -> LSTMCrisisPredictor:
        """Train LSTM model"""
        print("Training LSTM model...")
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_lstm_data(
            df, feature_cols, target_col
        )
        
        # Initialize model
        input_size = len(feature_cols)
        model = LSTMCrisisPredictor(
            input_size=input_size,
            hidden_size=self.config['LSTM']['hidden_size'],
            num_layers=self.config['LSTM']['num_layers'],
            dropout=self.config['LSTM']['dropout']
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config['LSTM']['learning_rate']
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            outputs, _ = model(X_train)
            loss = criterion(outputs, y_train)
            
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs, _ = model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
                print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
        
        print("✓ LSTM training completed")
        return model
    
    def train_xgboost(self, df: pd.DataFrame, feature_cols: List[str], 
                     target_col: str) -> XGBoostCrisisModel:
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Initialize model
        model = XGBoostCrisisModel(self.config['XGBOOST'])
        
        # Prepare data using the model's prepare_features method
        X = model.prepare_features(df, feature_cols)
        y = df[target_col].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        print("✓ XGBoost training completed")
        return model
