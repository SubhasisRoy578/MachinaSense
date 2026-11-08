import os
import sys
import psutil
import pandas as pd

p = psutil.Process()
def mem():
    return p.memory_info().rss / (1024 * 1024)

print(f"[1] Base Python Memory: {mem():.2f} MB")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import app.main
from app.main import app
from app.ml_service import ml_service, FEATURE_COLS, COLAB_FD001_FEATURE_ORDER

print(f"[2] After importing app.main (all routes & services): {mem():.2f} MB")

# Run lifespan artifact load
ml_service.load_artifacts()
print(f"[3] After ml_service.load_artifacts() (startup): {mem():.2f} MB")

# Test 1: Inference with Colab FD001 Random Forest (the 95MB model)
sample_colab_features = {
    "cycle": 1,
    "op_setting_1": -0.0007,
    "op_setting_2": -0.0004,
    "op_setting_3": 100.0,
    "sensor_2": 641.82,
    "sensor_3": 1589.70,
    "sensor_4": 1400.60,
    "sensor_6": 21.61,
    "sensor_7": 554.36,
    "sensor_8": 2388.06,
    "sensor_9": 9046.19,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.02,
    "sensor_14": 8138.62,
    "sensor_15": 8.4195,
    "sensor_17": 392.0,
    "sensor_20": 39.06,
    "sensor_21": 23.4190,
}
colab_pred = ml_service.predict_rul_colab(sample_colab_features)
print(f"[4] After Colab FD001 RF inference (pred={colab_pred}): {mem():.2f} MB")

# Test 2: Inference with Legacy Random Forest
sample_legacy_df = pd.DataFrame([{col: 100.0 for col in FEATURE_COLS}])
legacy_rf_pred = ml_service.predict_rul_raw(sample_legacy_df, use_lstm=False)
print(f"[5] After Legacy RF inference (pred={legacy_rf_pred}): {mem():.2f} MB")

# Test 3: Inference with Isolation Forest Anomaly Detector
anomaly_result = ml_service.detect_anomaly_raw(sample_legacy_df)
print(f"[6] After Isolation Forest inference (result={anomaly_result['severity']}): {mem():.2f} MB")

# Test 4: Inference with PyTorch LSTM (loads torch lazily)
sample_seq_df = pd.DataFrame([{col: 100.0 for col in FEATURE_COLS}] * 35)
lstm_pred = ml_service.predict_rul_raw(sample_seq_df, use_lstm=True)
print(f"[7] After PyTorch LSTM inference (pred={lstm_pred}): {mem():.2f} MB")

print(f"\nFinal Total Active Memory: {mem():.2f} MB")
print(f"Memory Budget: 512.00 MB")
print(f"Available Headroom: {512.00 - mem():.2f} MB")
