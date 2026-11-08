import os
import sys
import psutil

process = psutil.Process()

def get_mem():
    return process.memory_info().rss / (1024 * 1024)

print(f"[0] Python Base Memory: {get_mem():.2f} MB")

# Measure individual heavy library imports
import torch
print(f"[1] After import torch: {get_mem():.2f} MB")

import sklearn
print(f"[2] After import sklearn: {get_mem():.2f} MB")

import pandas as pd
print(f"[3] After import pandas: {get_mem():.2f} MB")

import numpy as np
print(f"[4] After import numpy: {get_mem():.2f} MB")

import sqlalchemy
print(f"[5] After import sqlalchemy: {get_mem():.2f} MB")

import fastapi
print(f"[6] After import fastapi: {get_mem():.2f} MB")

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import config & db
from app.config import settings
from app.db.session import init_db, check_db_ping
print(f"[7] After import config & db session: {get_mem():.2f} MB")

# Import ml_service (before loading artifacts)
from app.ml_service import ml_service, ARTIFACTS_DIR, COLAB_ARTIFACTS_DIR, DATA_DIR
print(f"[8] After import ml_service: {get_mem():.2f} MB")

# Import app.main
from app.main import app
print(f"[9] After import app.main (all routes/services imported): {get_mem():.2f} MB")

# Now let's measure individual model loading in ml_service
import joblib

scaler_path = os.path.join(ARTIFACTS_DIR, "preprocessing", "scaler.joblib")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f"[10a] Loaded scaler.joblib: {get_mem():.2f} MB")

rf_path = os.path.join(ARTIFACTS_DIR, "rul", "rf_baseline.joblib")
if os.path.exists(rf_path):
    rf = joblib.load(rf_path)
    print(f"[10b] Loaded legacy rf_baseline.joblib (~12MB file): {get_mem():.2f} MB")

lstm_path = os.path.join(ARTIFACTS_DIR, "rul", "lstm_model.pth")
if os.path.exists(lstm_path):
    from ml.models.lstm import RULLSTM, load_lstm_model
    from ml.data.loader import FEATURE_COLS
    lstm = RULLSTM(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2)
    lstm = load_lstm_model(lstm, lstm_path, device="cpu")
    print(f"[10c] Loaded legacy lstm_model.pth: {get_mem():.2f} MB")

anomaly_path = os.path.join(ARTIFACTS_DIR, "anomaly", "isolation_forest.joblib")
if os.path.exists(anomaly_path):
    anom = joblib.load(anomaly_path)
    print(f"[10d] Loaded isolation_forest.joblib: {get_mem():.2f} MB")

# Check CMAPSS raw dataset loading
if os.path.exists(DATA_DIR):
    from ml.data.loader import load_cmapss_data
    train_df, test_df, true_rul_df = load_cmapss_data(DATA_DIR, "FD001")
    print(f"[10e] Loaded CMAPSS raw data (train_df, test_df): {get_mem():.2f} MB")
    scaled_test = scaler.transform(test_df[FEATURE_COLS])
    print(f"[10f] Computed scaled_test_df: {get_mem():.2f} MB")

# Colab FD001 model loading (95MB on disk!)
colab_rf_path = os.path.join(COLAB_ARTIFACTS_DIR, "rf_baseline.joblib")
if os.path.exists(colab_rf_path):
    colab_rf = joblib.load(colab_rf_path)
    print(f"[10g] Loaded Colab 95MB rf_baseline.joblib: {get_mem():.2f} MB")

print(f"\nTOTAL PEAK MEMORY: {get_mem():.2f} MB")
