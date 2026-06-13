"""
Model Ensemble
Combines outputs of multiple models to generate directional probability and confidence.
"""
import os
import sys
import joblib
import torch
import numpy as np
import pandas as pd
from loguru import logger

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMClassifier

class EnsemblePredictor:
    def __init__(self):
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved')
        
        # Load tree models and scaler
        self.xgb = joblib.load(os.path.join(models_dir, 'xgb.pkl'))
        self.lgb = joblib.load(os.path.join(models_dir, 'lgb.pkl'))
        self.rf = joblib.load(os.path.join(models_dir, 'rf.pkl'))
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        
        # Load PyTorch LSTM
        self.lstm = LSTMClassifier(input_dim=6, hidden_dim=64, num_layers=2)
        lstm_path = os.path.join(models_dir, 'lstm.pt')
        self.lstm.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
        self.lstm.eval()
        
        logger.info("EnsemblePredictor successfully loaded all models (XGBoost, LightGBM, Random Forest, LSTM).")

    def predict(self, features_df: pd.DataFrame) -> dict:
        """
        Runs all models on the technical features dataframe and returns combined predictions.
        
        Args:
            features_df: DataFrame containing at least 20 rows of calculated technical indicators.
        """
        if len(features_df) < 20:
            raise ValueError(f"EnsemblePredictor.predict requires at least 20 rows of history (received {len(features_df)})")
            
        feature_cols = ['RSI_14', 'BB_Width', 'Return_1d', 'Return_5d', 'Volume_Ratio', 'HV_10']
        
        # Standardize columns to titles
        features_df = features_df.copy()
        features_df.columns = [col.title() for col in features_df.columns]
        
        # Ensure column names map correctly to expected case
        df_sub = pd.DataFrame()
        for col in feature_cols:
            title_col = col.title()
            if title_col in features_df.columns:
                df_sub[col] = features_df[title_col]
            else:
                logger.error(f"Required feature '{col}' ({title_col}) not found in features dataframe.")
                df_sub[col] = 0.0
                
        # Scale features
        scaled_values = self.scaler.transform(df_sub)
        
        # Tree model input (latest row)
        X_tree = scaled_values[-1:]
        
        # LSTM model input (last 20 rows)
        X_lstm = np.expand_dims(scaled_values[-20:], axis=0)
        X_lstm_tensor = torch.tensor(X_lstm, dtype=torch.float32)
        
        # Run Tree Models
        p_xgb = float(self.xgb.predict_proba(X_tree)[0, 1])
        p_lgb = float(self.lgb.predict_proba(X_tree)[0, 1])
        p_rf = float(self.rf.predict_proba(X_tree)[0, 1])
        
        # Run LSTM Model
        with torch.no_grad():
            p_lstm = float(self.lstm(X_lstm_tensor).item())
            
        # Combine: final_prob = 0.35*xgb + 0.35*lgb + 0.20*rf + 0.10*lstm
        final_prob = 0.35 * p_xgb + 0.35 * p_lgb + 0.20 * p_rf + 0.10 * p_lstm
        
        # Determine direction
        if final_prob >= 0.55:
            direction = "Bullish"
        elif final_prob <= 0.45:
            direction = "Bearish"
        else:
            direction = "Neutral"
            
        confidence = float(abs(final_prob - 0.5) * 2.0)
        
        # 5-day rolling HV projection
        if 'Hv_5' in features_df.columns:
            volatility_forecast = float(features_df['Hv_5'].iloc[-5:].mean())
        else:
            volatility_forecast = float(features_df['Return_1d'].iloc[-5:].std() * np.sqrt(252))
            
        return {
            'direction': direction,
            'probability': final_prob,
            'confidence': confidence,
            'volatility_forecast': volatility_forecast,
            'model_scores': {
                'xgboost': p_xgb,
                'lightgbm': p_lgb,
                'random_forest': p_rf,
                'lstm': p_lstm
            }
        }
