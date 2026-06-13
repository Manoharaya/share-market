"""
Model Training Script
Trains XGBoost, Random Forest, LightGBM, and PyTorch LSTM classifiers on historical NIFTY features.
Saves models to models/saved/
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from loguru import logger
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.technical import TechnicalFeatures
from models.lstm_model import LSTMClassifier

def create_sequences(X, y, lookback=20):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback + 1):
        X_seq.append(X.iloc[i : i + lookback].values)
        y_seq.append(y.iloc[i + lookback - 1])
    return np.array(X_seq), np.array(y_seq)

def train_models():
    os.makedirs('models/saved', exist_ok=True)
    
    # 1. Load historical daily NIFTY data
    csv_path = 'data/cache/NIFTY_daily.csv'
    if not os.path.exists(csv_path):
        logger.error(f"Cache file {csv_path} not found. Run historical caching first.")
        return
        
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Calculate features
    logger.info("Computing technical features...")
    df_feat = TechnicalFeatures().compute(df)
    
    # 2. Create labels (5-day 0.5% threshold)
    logger.info("Generating classification labels...")
    df_feat['label'] = (df_feat['Close'].shift(-5) > df_feat['Close'] * 1.005).astype(int)
    
    # Drop rows with NaN (first 200 due to EMA_200, last 5 due to shift)
    df_clean = df_feat.dropna().copy()
    
    # Define features and target
    feature_cols = ['RSI_14', 'BB_Width', 'Return_1d', 'Return_5d', 'Volume_Ratio', 'HV_10']
    
    X = df_clean[feature_cols]
    y = df_clean['label']
    
    logger.info(f"Total clean samples for training: {len(X)}. Total features: {len(feature_cols)}")
    
    # 3. Train/Test Split (last 20% as test, time-series split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features (critical for LSTM Convergence)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
    
    # Save scaler for runtime use
    joblib.dump(scaler, 'models/saved/scaler.pkl')
    
    # ----------------------------------------------------
    # Model 1: XGBoost
    # ----------------------------------------------------
    logger.info("Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, 
        max_depth=6, 
        learning_rate=0.05, 
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # ----------------------------------------------------
    # Model 2: Random Forest
    # ----------------------------------------------------
    logger.info("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=8, 
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # ----------------------------------------------------
    # Model 3: LightGBM
    # ----------------------------------------------------
    logger.info("Training LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)

    # Evaluate tree models
    for name, model in [('XGBoost', xgb_model), ('Random Forest', rf_model), ('LightGBM', lgb_model)]:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        
        logger.info(f"--- {name} Test Performance ---")
        logger.info(f"Accuracy:  {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall:    {rec:.4f}")
        logger.info(f"AUC-ROC:   {auc:.4f}")
        
        # Save model
        filename = f"models/saved/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        
        # Short alias
        short_name = "xgb.pkl" if name == "XGBoost" else "rf.pkl" if name == "Random Forest" else "lgb.pkl"
        joblib.dump(model, f"models/saved/{short_name}")
        logger.info(f"Saved {name} to {filename} and models/saved/{short_name}")

    # ----------------------------------------------------
    # Model 4: PyTorch LSTM
    # ----------------------------------------------------
    logger.info("Preparing sequence data for LSTM (20-day lookback)...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback=20)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback=20)
    
    logger.info(f"LSTM Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")
    
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1)
    
    lstm_model = LSTMClassifier(input_dim=len(feature_cols), hidden_dim=64, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
    
    logger.info("Training LSTM Classifier for 30 epochs...")
    lstm_model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = lstm_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/30 | Loss: {loss.item():.4f}")
            
    # Evaluate LSTM
    lstm_model.eval()
    with torch.no_grad():
        test_probs = lstm_model(X_test_tensor).numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        
    acc = accuracy_score(y_test_seq, test_preds)
    prec = precision_score(y_test_seq, test_preds, zero_division=0)
    rec = recall_score(y_test_seq, test_preds, zero_division=0)
    auc = roc_auc_score(y_test_seq, test_probs)
    
    logger.info(f"--- LSTM Test Performance ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"AUC-ROC:   {auc:.4f}")
    
    # Save LSTM model
    torch.save(lstm_model.state_dict(), 'models/saved/lstm.pt')
    logger.info("Saved LSTM weights to models/saved/lstm.pt")

if __name__ == '__main__':
    train_models()
