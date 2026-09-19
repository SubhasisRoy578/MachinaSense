import os
import sys
import json
import random
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Fix random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Adjust path to import ml modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ml.data.loader import load_cmapss_data, add_rul_labels, generate_sequences, generate_test_sequences, FEATURE_COLS
from ml.preprocessing.scaler import SensorScaler
from ml.models.baseline import BaselineRULModel
from ml.models.lstm import RULLSTM, save_lstm_model
from ml.models.anomaly import AnomalyDetector

# Config
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "ml", "data", "raw"))
ARTIFACTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
SEQ_LENGTH = 30
MAX_RUL = 125

def train_and_evaluate():
    set_seed(42)
    
    # Ensure artifact subdirectories exist
    os.makedirs(os.path.join(ARTIFACTS_DIR, "rul"), exist_ok=True)
    os.makedirs(os.path.join(ARTIFACTS_DIR, "anomaly"), exist_ok=True)
    os.makedirs(os.path.join(ARTIFACTS_DIR, "preprocessing"), exist_ok=True)

    print("Loading Data...")
    train_df, test_df, true_rul_df = load_cmapss_data(DATA_DIR, "FD001")
    
    print("Preprocessing...")
    scaler = SensorScaler(FEATURE_COLS)
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)
    
    # Save scaler artifact
    scaler.save(os.path.join(ARTIFACTS_DIR, "preprocessing", "scaler.joblib"))
    
    train_df = add_rul_labels(train_df, max_rul=MAX_RUL)
    
    # Unit-aware train / validation split for internal model evaluation (80% train engines, 20% val engines)
    unique_engines = train_df['id'].unique()
    np.random.shuffle(unique_engines)
    train_engine_count = int(0.8 * len(unique_engines))
    train_engine_ids = set(unique_engines[:train_engine_count])
    val_engine_ids = set(unique_engines[train_engine_count:])
    
    train_split_df = train_df[train_df['id'].isin(train_engine_ids)].copy()
    val_split_df = train_df[train_df['id'].isin(val_engine_ids)].copy()
    
    # ---------------------------
    # 1. Anomaly Detection (Isolation Forest)
    # ---------------------------
    print("Training Anomaly Detector (Isolation Forest - Unsupervised)...")
    # Train on "healthy" early-life engine data (RUL > 100)
    healthy_data = train_df[train_df['RUL'] > 100][FEATURE_COLS]
    anomaly_detector = AnomalyDetector(contamination=0.05, random_state=42)
    anomaly_detector.fit(healthy_data)
    anomaly_detector.save(os.path.join(ARTIFACTS_DIR, "anomaly", "isolation_forest.joblib"))
    
    # ---------------------------
    # 2. Baseline RUL (Random Forest)
    # ---------------------------
    print("Training Random Forest Baseline...")
    X_train_rf = train_df[FEATURE_COLS]
    y_train_rf = train_df['RUL']
    
    rf_model = BaselineRULModel(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)
    rf_model.save(os.path.join(ARTIFACTS_DIR, "rul", "rf_baseline.joblib"))
    
    # Evaluate RF Baseline on NASA Test Set
    last_cycle_test = test_df.groupby('id').last().reset_index()
    X_test_rf = last_cycle_test[FEATURE_COLS]
    y_pred_rf = rf_model.predict(X_test_rf)
    
    y_true = true_rul_df['RUL'].values
    
    rf_mae = float(mean_absolute_error(y_true, y_pred_rf))
    rf_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_rf)))
    rf_r2 = float(r2_score(y_true, y_pred_rf))
    print(f"RF Baseline Test Results -> MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, R2: {rf_r2:.2f}")

    # ---------------------------
    # 3. Temporal RUL (PyTorch LSTM)
    # ---------------------------
    print("Preparing Sequences for PyTorch LSTM...")
    X_train_seq, y_train_seq = generate_sequences(train_df, SEQ_LENGTH, FEATURE_COLS)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32), 
        torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    lstm_model = RULLSTM(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    epochs = 20
    print(f"Training PyTorch LSTM for {epochs} epochs...")
    for epoch in range(epochs):
        lstm_model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            
    save_lstm_model(lstm_model, os.path.join(ARTIFACTS_DIR, "rul", "lstm_model.pth"))
    
    # Evaluate LSTM on NASA Test Set
    lstm_model.eval()
    X_test_seq, test_engine_ids = generate_test_sequences(test_df, SEQ_LENGTH, FEATURE_COLS)
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        y_pred_lstm = lstm_model(X_test_tensor).cpu().numpy().flatten()
    
    # Align predictions with true RUL using engine ids
    true_rul_dict = dict(zip(true_rul_df['id'], true_rul_df['RUL']))
    y_true_lstm = [true_rul_dict[eid] for eid in test_engine_ids]
    
    lstm_mae = float(mean_absolute_error(y_true_lstm, y_pred_lstm))
    lstm_rmse = float(np.sqrt(mean_squared_error(y_true_lstm, y_pred_lstm)))
    lstm_r2 = float(r2_score(y_true_lstm, y_pred_lstm))
    print(f"PyTorch LSTM Test Results -> MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}, R2: {lstm_r2:.2f}")

    # ---------------------------
    # 4. Save Metadata & Evaluation
    # ---------------------------
    metadata = {
        "dataset": "C-MAPSS FD001",
        "training_date": datetime.now().isoformat(),
        "random_seed": 42,
        "models": {
            "baseline": "Random Forest Regressor (n_estimators=100, max_depth=10)",
            "temporal": "PyTorch LSTM (hidden_size=64, num_layers=2, dropout=0.2)",
            "anomaly": "Isolation Forest (contamination=0.05)"
        },
        "features": FEATURE_COLS,
        "preprocessing": "MinMaxScaler",
        "window_size": SEQ_LENGTH,
        "rul_cap": MAX_RUL
    }
    
    evaluation = {
        "rul_models": {
            "random_forest": {
                "mae": round(rf_mae, 4),
                "rmse": round(rf_rmse, 4),
                "r2": round(rf_r2, 4)
            },
            "lstm": {
                "mae": round(lstm_mae, 4),
                "rmse": round(lstm_rmse, 4),
                "r2": round(lstm_r2, 4)
            }
        },
        "anomaly_detection": {
            "method": "Isolation Forest",
            "scoring": "Inverted decision function (higher = more anomalous)",
            "severity_thresholds": {
                "healthy": "<= 0.50",
                "medium": "> 0.50",
                "high": "> 0.60",
                "critical": "> 0.75"
            },
            "note": "Unsupervised anomaly detection trained on healthy operational cycles (RUL > 100). FD001 dataset has no ground-truth anomaly labels."
        }
    }
    
    with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    with open(os.path.join(ARTIFACTS_DIR, "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
        
    print("Training pipeline completed successfully.")
    print(f"Artifacts saved in: {os.path.abspath(ARTIFACTS_DIR)}")

if __name__ == "__main__":
    train_and_evaluate()

