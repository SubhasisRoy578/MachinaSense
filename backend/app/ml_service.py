import os
import sys
import json
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Ensure project root and backend are in sys.path
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.models.lstm import RULLSTM, load_lstm_model
from ml.data.loader import load_cmapss_data, FEATURE_COLS, MAX_RUL, COLUMNS

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "ml", "data", "raw")

SENSOR_NAMES_MAP = {
    's2': {'name': 'LPC Outlet Temp', 'unit': '°R', 'min': 640.0, 'max': 650.0},
    's3': {'name': 'HPC Outlet Temp', 'unit': '°R', 'min': 1580.0, 'max': 1610.0},
    's4': {'name': 'LPT Outlet Temp', 'unit': '°R', 'min': 1400.0, 'max': 1430.0},
    's6': {'name': 'Bypass Duct Pressure', 'unit': 'psia', 'min': 21.0, 'max': 21.7},
    's7': {'name': 'HPC Outlet Pressure', 'unit': 'psia', 'min': 550.0, 'max': 556.0},
    's8': {'name': 'Physical Fan Speed', 'unit': 'rpm', 'min': 2387.0, 'max': 2389.0},
    's9': {'name': 'Physical Core Speed', 'unit': 'rpm', 'min': 9040.0, 'max': 9075.0},
    's11': {'name': 'HPC Static Pressure', 'unit': 'psia', 'min': 47.0, 'max': 48.5},
    's12': {'name': 'Fuel Flow Ratio', 'unit': 'pps/psi', 'min': 518.0, 'max': 523.0},
    's13': {'name': 'Corrected Fan Speed', 'unit': 'rpm', 'min': 2387.0, 'max': 2389.0},
    's14': {'name': 'Corrected Core Speed', 'unit': 'rpm', 'min': 8100.0, 'max': 8250.0},
    's15': {'name': 'Bypass Ratio', 'unit': '--', 'min': 8.3, 'max': 8.5},
    's17': {'name': 'Bleed Enthalpy', 'unit': 'BTU/lb', 'min': 390.0, 'max': 396.0},
    's20': {'name': 'HPT Cool Bleed', 'unit': 'lbm/s', 'min': 38.5, 'max': 39.0},
    's21': {'name': 'LPT Cool Bleed', 'unit': 'lbm/s', 'min': 23.0, 'max': 23.5},
}

class MLService:
    def __init__(self):
        self.scaler = None
        self.rf_baseline = None
        self.lstm_model = None
        self.anomaly_detector = None
        self.metadata: Dict[str, Any] = {}
        self.evaluation: Dict[str, Any] = {}
        self.is_loaded = False
        
        self.test_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.true_rul_df: Optional[pd.DataFrame] = None
        self.scaled_test_df: Optional[np.ndarray] = None

    def load_artifacts(self):
        try:
            scaler_path = os.path.join(ARTIFACTS_DIR, "preprocessing", "scaler.joblib")
            rf_path = os.path.join(ARTIFACTS_DIR, "rul", "rf_baseline.joblib")
            lstm_path = os.path.join(ARTIFACTS_DIR, "rul", "lstm_model.pth")
            anomaly_path = os.path.join(ARTIFACTS_DIR, "anomaly", "isolation_forest.joblib")
            meta_path = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
            eval_path = os.path.join(ARTIFACTS_DIR, "evaluation.json")

            if not (os.path.exists(scaler_path) and os.path.exists(rf_path) and 
                    os.path.exists(lstm_path) and os.path.exists(anomaly_path)):
                print("ML Service Warning: Trained artifact files missing. Please run train.py first.")
                self.is_loaded = False
                return False

            self.scaler = joblib.load(scaler_path)
            self.rf_baseline = joblib.load(rf_path)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.lstm_model = RULLSTM(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2).to(device)
            self.lstm_model = load_lstm_model(self.lstm_model, lstm_path, device=device)
            
            self.anomaly_detector = joblib.load(anomaly_path)
            
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.metadata = json.load(f)
                    
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    self.evaluation = json.load(f)
                    
            # Load dataset for querying machine endpoints
            if os.path.exists(DATA_DIR):
                try:
                    self.train_df, self.test_df, self.true_rul_df = load_cmapss_data(DATA_DIR, "FD001")
                    self.scaled_test_df = self.scaler.transform(self.test_df[FEATURE_COLS])
                except Exception as dataset_err:
                    print(f"Warning loading CMAPSS raw dataset: {dataset_err}")

            self.is_loaded = True
            print("MLService successfully initialized and all artifacts loaded.")
            return True
        except Exception as e:
            print(f"Error loading ML artifacts in MLService: {e}")
            self.is_loaded = False
            return False

    def parse_machine_id(self, machine_id_str: str) -> Optional[int]:
        clean_str = str(machine_id_str).strip()
        if clean_str.isdigit():
            return int(clean_str)
        if '-' in clean_str:
            parts = clean_str.split('-')
            if parts[-1].isdigit():
                return int(parts[-1])
        digits = ''.join(filter(str.isdigit, clean_str))
        if digits:
            return int(digits)
        return None

    def predict_rul_raw(self, features_df: pd.DataFrame, use_lstm: bool = True) -> float:
        if not self.is_loaded:
            raise ValueError("ML models are not loaded.")

        if all(col in features_df.columns for col in FEATURE_COLS):
            input_df = features_df[FEATURE_COLS]
        else:
            input_df = features_df

        scaled_arr = self.scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_arr, columns=FEATURE_COLS)
        seq_len = self.metadata.get("window_size", 30)

        if use_lstm and len(scaled_df) >= seq_len:
            seq = scaled_df.iloc[-seq_len:].values
            seq = np.expand_dims(seq, axis=0)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.no_grad():
                tensor = torch.tensor(seq, dtype=torch.float32).to(device)
                pred = self.lstm_model(tensor).cpu().numpy().flatten()[0]
            return max(0.0, float(pred))
        else:
            last_row = scaled_df.iloc[-1:]
            pred = self.rf_baseline.predict(last_row)[0]
            return max(0.0, float(pred))

    def detect_anomaly_raw(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_loaded:
            raise ValueError("ML models are not loaded.")

        if all(col in features_df.columns for col in FEATURE_COLS):
            input_df = features_df[FEATURE_COLS]
        else:
            input_df = features_df

        scaled_arr = self.scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_arr, columns=FEATURE_COLS)
        last_row = scaled_df.iloc[-1:]

        raw_score = self.anomaly_detector.score_samples(last_row)[0]
        anomaly_score = float(-raw_score)
        normalized_score = float(np.clip((anomaly_score + 0.1) / 0.45, 0.0, 1.0))

        if normalized_score > 0.75:
            severity = 'critical'
        elif normalized_score > 0.60:
            severity = 'high'
        elif normalized_score > 0.45:
            severity = 'medium'
        else:
            severity = 'healthy'

        return {
            "raw_score": anomaly_score,
            "anomaly_score": round(normalized_score, 4),
            "severity": severity,
            "is_anomaly": normalized_score > 0.55
        }

    def get_machine_list(self) -> List[Dict[str, Any]]:
        if self.test_df is None or not self.is_loaded:
            return []

        machines = []
        for engine_id in sorted(self.test_df['id'].unique()):
            engine_data = self.test_df[self.test_df['id'] == engine_id]
            max_cycle = int(engine_data['cycle'].max())
            
            rul_pred = self.predict_rul_raw(engine_data, use_lstm=True)
            rf_pred = self.predict_rul_raw(engine_data, use_lstm=False)
            anomaly_info = self.detect_anomaly_raw(engine_data)

            health_score = int(np.clip((rul_pred / MAX_RUL) * 100, 0, 100))
            
            if health_score > 65:
                status = 'operating'
            elif health_score > 30:
                status = 'degraded'
            else:
                status = 'critical'

            machines.append({
                "id": f"FD001-{engine_id:03d}",
                "numericId": engine_id,
                "name": f"Turbofan Engine FD001-{engine_id:03d}",
                "model": "C-MAPSS FD001 Turbofan",
                "location": f"Test Bench {(engine_id % 5) + 1}",
                "status": status,
                "runningHours": max_cycle * 2,
                "currentCycle": max_cycle,
                "healthScore": health_score,
                "predictedRul": round(rul_pred, 1),
                "rfPredictedRul": round(rf_pred, 1),
                "anomalySeverity": anomaly_info["severity"],
                "anomalyScore": anomaly_info["anomaly_score"],
                "lastMaintenance": f"Cycle {max(0, max_cycle - 30)}"
            })
        return machines

    def get_machine_detail(self, machine_id_str: str) -> Optional[Dict[str, Any]]:
        num_id = self.parse_machine_id(machine_id_str)
        if num_id is None or self.test_df is None:
            return None

        engine_data = self.test_df[self.test_df['id'] == num_id]
        if engine_data.empty:
            return None

        max_cycle = int(engine_data['cycle'].max())
        rul_pred = self.predict_rul_raw(engine_data, use_lstm=True)
        rf_pred = self.predict_rul_raw(engine_data, use_lstm=False)
        anomaly_info = self.detect_anomaly_raw(engine_data)
        health_score = int(np.clip((rul_pred / MAX_RUL) * 100, 0, 100))

        if health_score > 65:
            status = 'operating'
        elif health_score > 30:
            status = 'degraded'
        else:
            status = 'critical'

        latest_row = engine_data.iloc[-1]
        sensors = []
        for s_code in FEATURE_COLS:
            meta = SENSOR_NAMES_MAP.get(s_code, {'name': s_code, 'unit': '', 'min': 0, 'max': 100})
            val = float(latest_row[s_code])
            sensors.append({
                "id": s_code,
                "name": meta['name'],
                "value": round(val, 2),
                "unit": meta['unit'],
                "min": meta['min'],
                "max": meta['max'],
                "status": "normal" if val <= meta['max'] else "high"
            })

        return {
            "id": f"FD001-{num_id:03d}",
            "numericId": num_id,
            "name": f"Turbofan Engine FD001-{num_id:03d}",
            "model": "C-MAPSS FD001 Turbofan",
            "type": "Turbofan Jet Engine",
            "location": f"Test Bench {(num_id % 5) + 1}",
            "installationDate": "2023-01-15",
            "lastMaintenance": f"Cycle {max(0, max_cycle - 30)}",
            "status": status,
            "runningHours": max_cycle * 2,
            "currentCycle": max_cycle,
            "healthScore": health_score,
            "predictedRul": round(rul_pred, 1),
            "rfPredictedRul": round(rf_pred, 1),
            "anomalySeverity": anomaly_info["severity"],
            "anomalyScore": anomaly_info["anomaly_score"],
            "sensors": sensors
        }

    def get_machine_sensors(self, machine_id_str: str) -> Optional[Dict[str, Any]]:
        num_id = self.parse_machine_id(machine_id_str)
        if num_id is None or self.test_df is None:
            return None

        engine_data = self.test_df[self.test_df['id'] == num_id]
        if engine_data.empty:
            return None

        history = []
        for _, row in engine_data.iterrows():
            cycle_point = {"cycle": int(row['cycle'])}
            for s_code in FEATURE_COLS:
                cycle_point[s_code] = round(float(row[s_code]), 2)
            history.append(cycle_point)

        sensors_meta = [
            {
                "key": s_code,
                "name": SENSOR_NAMES_MAP[s_code]['name'],
                "unit": SENSOR_NAMES_MAP[s_code]['unit']
            }
            for s_code in FEATURE_COLS
        ]

        return {
            "machineId": f"FD001-{num_id:03d}",
            "sensors": sensors_meta,
            "history": history
        }

    def get_machine_prediction(self, machine_id_str: str) -> Optional[Dict[str, Any]]:
        num_id = self.parse_machine_id(machine_id_str)
        if num_id is None or self.test_df is None:
            return None

        engine_data = self.test_df[self.test_df['id'] == num_id]
        if engine_data.empty:
            return None

        lstm_rul = self.predict_rul_raw(engine_data, use_lstm=True)
        rf_rul = self.predict_rul_raw(engine_data, use_lstm=False)
        mae = self.evaluation.get("rul_models", {}).get("lstm", {}).get("mae", 11.3)

        health_score = int(np.clip((lstm_rul / MAX_RUL) * 100, 0, 100))

        if health_score > 65:
            risk_level = 'low'
        elif health_score > 40:
            risk_level = 'medium'
        elif health_score > 20:
            risk_level = 'high'
        else:
            risk_level = 'critical'

        max_c = int(engine_data['cycle'].max())
        degradation_curve = []
        step = max(1, max_c // 20)
        for c in range(1, max_c + 1, step):
            sub_data = engine_data[engine_data['cycle'] <= c]
            if len(sub_data) > 0:
                c_rul = self.predict_rul_raw(sub_data, use_lstm=True)
                c_health = int(np.clip((c_rul / MAX_RUL) * 100, 0, 100))
                degradation_curve.append({
                    "cycle": c,
                    "predictedRul": round(c_rul, 1),
                    "healthIndex": c_health
                })

        return {
            "machineId": f"FD001-{num_id:03d}",
            "currentCycle": max_c,
            "predictedRul": round(lstm_rul, 1),
            "rfPredictedRul": round(rf_rul, 1),
            "confidenceInterval": {
                "min": max(0.0, round(lstm_rul - mae, 1)),
                "max": round(lstm_rul + mae, 1)
            },
            "healthScore": health_score,
            "riskLevel": risk_level,
            "degradationCurve": degradation_curve,
            "recommendations": [
                "Schedule High-Pressure Compressor inspection within 15 cycles" if lstm_rul < 40 else "Routine monitoring sufficient",
                "Calibrate temperature sensors s2 and s4" if lstm_rul < 60 else "Sensor readings within nominal calibration bounds"
            ]
        }

    def get_machine_anomalies(self, machine_id_str: str) -> Optional[Dict[str, Any]]:
        num_id = self.parse_machine_id(machine_id_str)
        if num_id is None or self.test_df is None:
            return None

        engine_data = self.test_df[self.test_df['id'] == num_id]
        if engine_data.empty:
            return None

        scaled_arr = self.scaler.transform(engine_data[FEATURE_COLS])
        
        anomalies_timeline = []
        for idx, (_, row) in enumerate(engine_data.iterrows()):
            cycle = int(row['cycle'])
            row_df = pd.DataFrame(scaled_arr[idx:idx+1], columns=FEATURE_COLS)
            raw_score = self.anomaly_detector.score_samples(row_df)[0]
            anom_score = float(-raw_score)
            norm_score = float(np.clip((anom_score + 0.1) / 0.45, 0.0, 1.0))

            if norm_score > 0.50:
                if norm_score > 0.75:
                    sev = 'critical'
                elif norm_score > 0.60:
                    sev = 'high'
                else:
                    sev = 'medium'

                anomalies_timeline.append({
                    "cycle": cycle,
                    "score": round(norm_score, 4),
                    "severity": sev,
                    "status": "investigating" if sev in ['high', 'critical'] else "observed",
                    "description": f"Elevated anomaly score ({round(norm_score, 2)}) detected at cycle {cycle}"
                })

        return {
            "machineId": f"FD001-{num_id:03d}",
            "totalAnomalies": len(anomalies_timeline),
            "anomalyEvents": anomalies_timeline
        }

    def get_machine_health(self, machine_id_str: str) -> Optional[Dict[str, Any]]:
        num_id = self.parse_machine_id(machine_id_str)
        if num_id is None or self.test_df is None:
            return None

        engine_data = self.test_df[self.test_df['id'] == num_id]
        if engine_data.empty:
            return None

        lstm_rul = self.predict_rul_raw(engine_data, use_lstm=True)
        anom = self.detect_anomaly_raw(engine_data)
        health_score = int(np.clip((lstm_rul / MAX_RUL) * 100, 0, 100))

        return {
            "machineId": f"FD001-{num_id:03d}",
            "healthScore": health_score,
            "rul": round(lstm_rul, 1),
            "anomalyScore": anom["anomaly_score"],
            "anomalySeverity": anom["severity"],
            "overallStatus": "good" if health_score > 60 else ("degraded" if health_score > 30 else "critical")
        }

ml_service = MLService()
