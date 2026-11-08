import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

# Ensure backend root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.data.loader import COLUMNS, FEATURE_COLS
from ml.models.lstm import RULLSTM, load_lstm_model

DATA_DIR = os.path.join(BASE_DIR, "ml", "data", "raw")
COLAB_DIR = os.path.join(BASE_DIR, "artifacts", "colab_fd001")
PHASE_10_DIR = os.path.join(COLAB_DIR, "phase_10")
EXPLAIN_DIR = os.path.join(PHASE_10_DIR, "explainability")
os.makedirs(EXPLAIN_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. LOAD TEST DATA & MODELS
# ---------------------------------------------------------------------
test_path = os.path.join(DATA_DIR, "test_FD001.txt")
rul_path = os.path.join(DATA_DIR, "RUL_FD001.txt")

RAW_COLS = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
FEATURE_ORDER_19 = [
    'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7',
    'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13',
    'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

df_test = pd.read_csv(test_path, sep=r'\s+', header=None, names=RAW_COLS)
df_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['ground_truth_rul'])
y_true_capped = np.minimum(df_rul['ground_truth_rul'].values, 125.0)

# Last observed cycle per engine for RF
last_rows = df_test.groupby('engine_id').last().reset_index()
X_test_19 = pd.DataFrame(last_rows[FEATURE_ORDER_19].values, columns=FEATURE_ORDER_19)

# Load RF Model
rf_path = os.path.join(COLAB_DIR, "rf_baseline.joblib")
rf_model = joblib.load(rf_path)

# Load LSTM Model & Scaler
scaler_path = os.path.join(BASE_DIR, "artifacts", "preprocessing", "scaler.joblib")
lstm_path = os.path.join(BASE_DIR, "artifacts", "rul", "lstm_model.pth")
scaler = joblib.load(scaler_path)

lstm_model = RULLSTM(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2)
lstm_model = load_lstm_model(lstm_model, lstm_path, device='cpu')
lstm_model.eval()

# Build 3D sequences for LSTM (100 engines, 30 cycles, 15 sensors)
lstm_sequences = []
test_df_orig = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMNS)
for engine_id in range(1, 101):
    eng_df = test_df_orig[test_df_orig['id'] == engine_id]
    scaled_data = scaler.transform(eng_df[FEATURE_COLS])
    seq = scaled_data[-30:]
    lstm_sequences.append(seq)
X_test_lstm = np.array(lstm_sequences) # shape (100, 30, 15)

print(f"Data shapes loaded: X_test_19={X_test_19.shape}, X_test_lstm={X_test_lstm.shape}")

# ---------------------------------------------------------------------
# 2. RANDOM FOREST EXPLAINABILITY
# ---------------------------------------------------------------------
print("\n--- Computing Random Forest Explainability ---")

# A. Global Gini Feature Importance
gini_importances = rf_model.feature_importances_
rf_global_df = pd.DataFrame({
    'feature': FEATURE_ORDER_19,
    'gini_importance': gini_importances,
    'importance_percentage': (gini_importances / gini_importances.sum()) * 100.0
}).sort_values(by='gini_importance', ascending=False).reset_index(drop=True)
rf_global_df['rank'] = rf_global_df.index + 1

rf_global_csv = os.path.join(EXPLAIN_DIR, "rf_global_feature_importance.csv")
rf_global_df.to_csv(rf_global_csv, index=False)
# Also copy to root phase_10 dir for required deliverable path
rf_global_df.to_csv(os.path.join(PHASE_10_DIR, "rf_global_feature_importance.csv"), index=False)

# B. Permutation Feature Importance on official test set
perm_res = permutation_importance(
    rf_model, X_test_19, y_true_capped,
    n_repeats=30, random_state=42, scoring='neg_mean_absolute_error'
)
# Note: neg_mean_absolute_error decrease indicates MAE increase when shuffled
perm_mean = perm_res.importances_mean
perm_std = perm_res.importances_std

rf_perm_df = pd.DataFrame({
    'feature': FEATURE_ORDER_19,
    'mae_increase_when_permuted': perm_mean,
    'std_deviation': perm_std
}).sort_values(by='mae_increase_when_permuted', ascending=False).reset_index(drop=True)
rf_perm_df['rank'] = rf_perm_df.index + 1

rf_perm_csv = os.path.join(EXPLAIN_DIR, "rf_permutation_importance.csv")
rf_perm_df.to_csv(rf_perm_csv, index=False)
rf_perm_df.to_csv(os.path.join(PHASE_10_DIR, "rf_permutation_importance.csv"), index=False)

# C. Individual Engine Explanations (Perturbational Feature Attribution)
# We test 5 representative engines:
# Unit 34 (terminal failure, RUL 7), Unit 82 (terminal, RUL 9), Unit 77 (mid-life, RUL 34),
# Unit 45 (outlier, RUL 114), Unit 1 (nominal health, RUL 112)
sample_engines = [1, 34, 45, 77, 82]
engine_explanations = []

for uid in sample_engines:
    eng_row = X_test_19.iloc[uid - 1:uid]
    baseline_pred = float(np.clip(rf_model.predict(eng_row)[0], 0, 125))
    actual_rul = float(y_true_capped[uid - 1])
    
    # Measure feature sensitivity: perturbation of +/- 5% and +/- 10%
    feature_impacts = []
    for feat in FEATURE_ORDER_19:
        val = eng_row[feat].values[0]
        # Perturb +5%
        row_pos = eng_row.copy()
        delta = abs(val) * 0.05 if abs(val) > 1e-4 else 0.05
        row_pos[feat] = val + delta
        pred_pos = float(np.clip(rf_model.predict(row_pos)[0], 0, 125))
        
        # Perturb -5%
        row_neg = eng_row.copy()
        row_neg[feat] = val - delta
        pred_neg = float(np.clip(rf_model.predict(row_neg)[0], 0, 125))
        
        # Sensitivity = average absolute RUL response
        sensitivity = float((abs(pred_pos - baseline_pred) + abs(pred_neg - baseline_pred)) / 2.0)
        direction = "negative" if pred_pos < baseline_pred else "positive"
        
        feature_impacts.append({
            'feature': feat,
            'baseline_value': float(val),
            'sensitivity_cycles_per_5pct': round(sensitivity, 4),
            'effect_direction': direction
        })
    
    # Sort top 5 contributing features by perturbation sensitivity
    top5_feats = sorted(feature_impacts, key=lambda x: x['sensitivity_cycles_per_5pct'], reverse=True)[:5]
    
    engine_explanations.append({
        'engine_id': uid,
        'final_observed_cycle': int(eng_row['cycle'].values[0]),
        'actual_rul': actual_rul,
        'predicted_rul': round(baseline_pred, 2),
        'residual': round(baseline_pred - actual_rul, 2),
        'top_5_contributing_features': top5_feats,
        'all_feature_impacts': feature_impacts
    })

rf_explain_json = {
    'model_name': 'MachinaSense Random Forest RUL Predictor',
    'algorithm': 'RandomForestRegressor (200 Estimators)',
    'feature_count': 19,
    'explanation_methods': [
        'Gini Impurity (Mean Decrease in Impurity)',
        'Permutation Feature Importance on Official Test Set (30 repeats)',
        'Local Feature Perturbation Sensitivity (Delta RUL on +/- 5% disturbance)'
    ],
    'top_5_global_features_gini': rf_global_df.head(5).to_dict(orient='records'),
    'top_5_global_features_permutation': rf_perm_df.head(5).to_dict(orient='records'),
    'sample_engine_explanations': engine_explanations
}

with open(os.path.join(EXPLAIN_DIR, "rf_explainability.json"), "w", encoding="utf-8") as f:
    json.dump(rf_explain_json, f, indent=2)
with open(os.path.join(PHASE_10_DIR, "rf_explainability.json"), "w", encoding="utf-8") as f:
    json.dump(rf_explain_json, f, indent=2)

# Visualization: RF Feature Importance
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'Segoe UI, Helvetica, Arial'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

# Gini
y_pos = np.arange(len(rf_global_df))
ax1.barh(y_pos, rf_global_df['gini_importance'].values[::-1], color='#38bdf8', edgecolor='#0284c7', alpha=0.85)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(rf_global_df['feature'].values[::-1], fontsize=9, color='#cbd5e1')
ax1.set_xlabel('Gini Impurity Importance (MDI)', fontsize=11, color='#cbd5e1')
ax1.set_title('Random Forest: Global Gini Feature Importance', fontsize=12, fontweight='bold', color='#f8fafc', pad=10)
ax1.grid(True, linestyle=':', alpha=0.3, color='#64748b')

# Permutation
y_pos2 = np.arange(len(rf_perm_df))
ax2.barh(y_pos2, rf_perm_df['mae_increase_when_permuted'].values[::-1], xerr=rf_perm_df['std_deviation'].values[::-1],
         color='#10b981', edgecolor='#059669', alpha=0.85, capsize=3, ecolor='#a7f3d0')
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(rf_perm_df['feature'].values[::-1], fontsize=9, color='#cbd5e1')
ax2.set_xlabel('MAE Increase When Permuted (Cycles)', fontsize=11, color='#cbd5e1')
ax2.set_title('Random Forest: Test-Set Permutation Importance (N=100)', fontsize=12, fontweight='bold', color='#f8fafc', pad=10)
ax2.grid(True, linestyle=':', alpha=0.3, color='#64748b')

plt.tight_layout()
rf_plot_path = os.path.join(EXPLAIN_DIR, "rf_feature_importance.png")
plt.savefig(rf_plot_path)
plt.savefig(os.path.join(PHASE_10_DIR, "rf_feature_importance.png"))
plt.close()
print("Saved RF explainability artifacts & plot.")

# ---------------------------------------------------------------------
# 3. LSTM EXPLAINABILITY (Gradient Saliency & Temporal Attribution)
# ---------------------------------------------------------------------
print("\n--- Computing LSTM Explainability (Gradient Saliency & Temporal Sensitivity) ---")

# Saliency / Input x Gradient across all 100 test engines
# Tensor: (100, 30, 15)
inputs_tensor = torch.tensor(X_test_lstm, dtype=torch.float32, requires_grad=True)
outputs = lstm_model(inputs_tensor) # (100, 1)

# Backprop sum of predictions to get gradients w.r.t input sequences
outputs.sum().backward()
grads = inputs_tensor.grad.detach().numpy() # shape: (100, 30, 15)

# Input x Gradient saliency
saliency = np.abs(grads)
input_x_grad = np.abs(X_test_lstm * grads)

# 1. Sensor-wise importance: average across 100 engines and 30 time steps
sensor_importance_raw = np.mean(saliency, axis=(0, 1))
sensor_ixg_raw = np.mean(input_x_grad, axis=(0, 1))

sensor_importance_pct = (sensor_importance_raw / sensor_importance_raw.sum()) * 100.0
lstm_sensor_df = pd.DataFrame({
    'sensor': FEATURE_COLS,
    'gradient_saliency': sensor_importance_raw,
    'input_x_gradient': sensor_ixg_raw,
    'importance_percentage': sensor_importance_pct
}).sort_values(by='gradient_saliency', ascending=False).reset_index(drop=True)
lstm_sensor_df['rank'] = lstm_sensor_df.index + 1

lstm_sensor_csv = os.path.join(EXPLAIN_DIR, "lstm_sensor_importance.csv")
lstm_sensor_df.to_csv(lstm_sensor_csv, index=False)
lstm_sensor_df.to_csv(os.path.join(PHASE_10_DIR, "lstm_sensor_importance.csv"), index=False)

# 2. Temporal time-step importance: average across 100 engines and 15 sensors
temporal_importance_raw = np.mean(saliency, axis=(0, 2)) # shape: (30,)
temporal_ixg_raw = np.mean(input_x_grad, axis=(0, 2))
temporal_pct = (temporal_importance_raw / temporal_importance_raw.sum()) * 100.0

time_steps_relative = [f"T-{29 - t}" if t < 29 else "T (Current)" for t in range(30)]
time_steps_idx = list(range(1, 31))

lstm_temporal_df = pd.DataFrame({
    'time_step_index': time_steps_idx,
    'relative_cycle': time_steps_relative,
    'gradient_saliency': temporal_importance_raw,
    'input_x_gradient': temporal_ixg_raw,
    'importance_percentage': temporal_pct
})

lstm_temporal_csv = os.path.join(EXPLAIN_DIR, "lstm_temporal_importance.csv")
lstm_temporal_df.to_csv(lstm_temporal_csv, index=False)
lstm_temporal_df.to_csv(os.path.join(PHASE_10_DIR, "lstm_temporal_importance.csv"), index=False)

# 3. Sensor x Time Step Heatmap Matrix (30 time steps x 15 sensors)
heatmap_matrix = np.mean(saliency, axis=0) # shape (30, 15)

# 4. Engine-specific LSTM sensitivity for sample engines
lstm_engine_explanations = []
for uid in sample_engines:
    eng_seq = X_test_lstm[uid - 1:uid] # (1, 30, 15)
    t_in = torch.tensor(eng_seq, dtype=torch.float32, requires_grad=True)
    out = lstm_model(t_in)
    out.backward()
    e_grad = np.abs(t_in.grad.detach().numpy()[0]) # (30, 15)
    
    e_sensor_imp = np.mean(e_grad, axis=0)
    top_sensors = [
        {'sensor': FEATURE_COLS[i], 'attribution': round(float(e_sensor_imp[i]), 5)}
        for i in np.argsort(e_sensor_imp)[::-1][:5]
    ]
    
    e_temp_imp = np.mean(e_grad, axis=1) # (30,)
    top_time_steps = [
        {'time_step': time_steps_relative[i], 'attribution': round(float(e_temp_imp[i]), 5)}
        for i in np.argsort(e_temp_imp)[::-1][:5]
    ]
    
    baseline_pred = float(np.clip(out.item(), 0, 125))
    actual_rul = float(y_true_capped[uid - 1])
    
    lstm_engine_explanations.append({
        'engine_id': uid,
        'actual_rul': actual_rul,
        'predicted_rul': round(baseline_pred, 2),
        'residual': round(baseline_pred - actual_rul, 2),
        'top_contributing_sensors': top_sensors,
        'top_influential_time_steps': top_time_steps
    })

lstm_explain_json = {
    'model_name': 'LSTM Neural Network RUL Predictor',
    'architecture': '2-Layer Recurrent LSTM (hidden_size=64, input_size=15, window=30)',
    'explanation_methods': [
        'Input Gradient Saliency (first-order Taylor approximation |dY/dX|)',
        'Input x Gradient Attribution (|X * dY/dX|)',
        'Sensor-wise Temporal Sensitivity Breakdown'
    ],
    'sensor_importance_ranking': lstm_sensor_df.to_dict(orient='records'),
    'temporal_importance_ranking': lstm_temporal_df.to_dict(orient='records'),
    'sample_engine_explanations': lstm_engine_explanations,
    'methodology_limitations': (
        "Gradient saliency measures local first-order sensitivity around the input point. "
        "It does not assume or require attention layers. Highly non-linear interactions across saturated gates "
        "may not be completely captured by first-order derivatives alone."
    )
}

with open(os.path.join(EXPLAIN_DIR, "lstm_explainability.json"), "w", encoding="utf-8") as f:
    json.dump(lstm_explain_json, f, indent=2)
with open(os.path.join(PHASE_10_DIR, "lstm_explainability.json"), "w", encoding="utf-8") as f:
    json.dump(lstm_explain_json, f, indent=2)

# Visualization: LSTM Sensor Importance
plt.figure(figsize=(9, 6), dpi=300)
y_pos = np.arange(len(lstm_sensor_df))
plt.barh(y_pos, lstm_sensor_df['gradient_saliency'].values[::-1], color='#10b981', edgecolor='#059669', alpha=0.85)
plt.yticks(y_pos, lstm_sensor_df['sensor'].values[::-1], color='#cbd5e1')
plt.xlabel('Mean Gradient Saliency |dY/dX|', fontsize=11, color='#cbd5e1')
plt.title('LSTM: Sensor Attribution Across 100 Official Test Engines', fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
plt.grid(True, linestyle=':', alpha=0.3, color='#64748b')
plt.tight_layout()
lstm_sensor_plot = os.path.join(EXPLAIN_DIR, "lstm_sensor_importance.png")
plt.savefig(lstm_sensor_plot)
plt.savefig(os.path.join(PHASE_10_DIR, "lstm_sensor_importance.png"))
plt.close()

# Visualization: LSTM Temporal Importance (30 cycles)
plt.figure(figsize=(10, 5.5), dpi=300)
x_ticks = np.arange(1, 31)
plt.plot(x_ticks, lstm_temporal_df['gradient_saliency'].values, marker='o', color='#38bdf8', linewidth=2.5, markersize=5, label='Gradient Sensitivity')
plt.fill_between(x_ticks, 0, lstm_temporal_df['gradient_saliency'].values, color='#38bdf8', alpha=0.2)
plt.axvline(30, color='#f43f5e', linestyle='--', linewidth=1.5, label='T (Current / Final Cycle)')
plt.title('LSTM: Temporal Importance Across 30-Cycle Input Window', fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
plt.xlabel('Cycle Index within Window (1 = T-29, 30 = T Current)', fontsize=11, color='#cbd5e1')
plt.ylabel('Mean Gradient Saliency', fontsize=11, color='#cbd5e1')
plt.xticks(x_ticks[::2], [f"T-{29 - (t-1)}" for t in x_ticks[::2]], color='#cbd5e1', rotation=45)
plt.grid(True, linestyle=':', alpha=0.3, color='#64748b')
plt.legend(frameon=True, facecolor='#0f172a', edgecolor='#334155')
plt.tight_layout()
lstm_temp_plot = os.path.join(EXPLAIN_DIR, "lstm_temporal_importance.png")
plt.savefig(lstm_temp_plot)
plt.savefig(os.path.join(PHASE_10_DIR, "lstm_temporal_importance.png"))
plt.close()
print("Saved LSTM explainability artifacts & plots.")

