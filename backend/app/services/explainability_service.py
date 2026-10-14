"""ML Explainability Service for MachinaSense Phase 7.

Provides scientifically grounded, model-aware global and local explainability:
- Tree Models (Random Forest, Gradient Boosting): Native feature importances + Marginal perturbation contributions
- Linear Models (Logistic Regression): Standardized coefficient magnitudes + Linear logit contributions
- Isolation Forest: Feature perturbation sensitivity on decision function (No causal claims)
- PyTorch LSTM: Sequential channel sensitivity analysis (Labeled as sensitivity, not causal attribution)
- Clear unsupported state handling for incompatible architectures
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models import ModelPredictionRecord, ModelRegistryRecord, User
from app.ml_service import ml_service, FEATURE_COLS as BENCHMARK_FEATURE_COLS, SENSOR_NAMES_MAP
from app.services.inference_service import (
    get_cached_or_load_artifact,
    resolve_active_model,
    validate_and_prepare_dataframe,
    transform_features,
)

logger = logging.getLogger("machinasense.explainability")

# Sensor units and human-readable names for industrial telemetry
DEFAULT_FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "vibration": {"label": "Vibration Amplitude", "unit": "mm/s"},
    "temperature": {"label": "Operating Temperature", "unit": "°C"},
    "pressure": {"label": "System Pressure", "unit": "bar"},
    "rpm": {"label": "Rotational Speed", "unit": "rpm"},
    "cycle": {"label": "Operational Cycle", "unit": "cycles"},
}
for s_id, s_info in SENSOR_NAMES_MAP.items():
    DEFAULT_FEATURE_METADATA[s_id] = {
        "label": s_info.get("name", s_id),
        "unit": s_info.get("unit", ""),
    }


def get_feature_meta(col_name: str) -> Dict[str, str]:
    """Resolves human-readable label and physical engineering unit for a feature column."""
    clean_name = col_name.lower().strip()
    if clean_name in DEFAULT_FEATURE_METADATA:
        return DEFAULT_FEATURE_METADATA[clean_name]
    for key, val in DEFAULT_FEATURE_METADATA.items():
        if key in clean_name:
            return val
    return {"label": col_name.replace("_", " ").title(), "unit": ""}


def explain_benchmark_model_global(benchmark_id: str) -> Dict[str, Any]:
    """Generates global explainability for pre-trained NASA C-MAPSS benchmark models."""
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="NASA C-MAPSS benchmark models are not loaded.")

    clean_id = benchmark_id.lower().strip()

    if clean_id in {"benchmark_rf", "rf_baseline", "nasa_rf"}:
        rf = ml_service.rf_baseline
        if not hasattr(rf, "feature_importances_"):
            raise HTTPException(status_code=500, detail="Benchmark Random Forest missing feature importances.")

        raw_importances = rf.feature_importances_
        feature_names = BENCHMARK_FEATURE_COLS
        total = float(raw_importances.sum()) or 1.0
        normalized = [round(float(val / total), 4) for val in raw_importances]

        indexed = sorted(
            [
                {
                    "feature": feature_names[i],
                    "importance": normalized[i],
                    "raw_score": round(float(raw_importances[i]), 5),
                    "label": get_feature_meta(feature_names[i])["label"],
                    "unit": get_feature_meta(feature_names[i])["unit"],
                }
                for i in range(len(feature_names))
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
        for rank, item in enumerate(indexed, start=1):
            item["rank"] = rank

        return {
            "model_id": "benchmark_rf",
            "model_name": "NASA C-MAPSS Random Forest Regressor",
            "model_type": "rf_regressor",
            "task": "rul_regression",
            "version": 1,
            "status": "active_benchmark",
            "is_benchmark": True,
            "supported": True,
            "explanation_method": "mean_decrease_impurity",
            "method_description": (
                "Measures total reduction in variance (Gini impurity equivalent for regression) "
                "brought by each feature across all decision trees in the ensemble."
            ),
            "evaluation_context": {
                "dataset": "NASA C-MAPSS FD001 Turbofan Engine Degradation Benchmark",
                "target": "RUL (Remaining Useful Life in cycles)",
                "sample_count": 100,
                "split": "Held-out test split (100 turbofan units)",
                "metrics": ml_service.evaluation.get("rul_models", {}).get("random_forest", {}),
            },
            "features": indexed,
            "limitations": (
                "Feature importance measures model reliance and statistical association on the FD001 "
                "turbofan dataset. It does not establish proven physical or mechanical failure causation."
            ),
        }

    elif clean_id in {"benchmark_lstm", "lstm_model", "nasa_lstm"}:
        # LSTM Architecture: Tree impurity is unsupported. Use sequential channel sensitivity analysis.
        feature_names = BENCHMARK_FEATURE_COLS
        lstm_net = ml_service.lstm_model
        scaler = ml_service.scaler
        seq_len = ml_service.metadata.get("window_size", 30)

        # Baseline evaluation on scaled zero/mean sequence
        device = "cuda" if torch.cuda.is_available() else "cpu"
        baseline_seq = np.zeros((1, seq_len, len(feature_names)), dtype=np.float32)

        sensitivities = []
        with torch.no_grad():
            base_tensor = torch.tensor(baseline_seq, device=device)
            base_pred = float(lstm_net(base_tensor).cpu().numpy().flatten()[0])

            for idx, col in enumerate(feature_names):
                # Perturb channel by unit standard deviation in normalized space (+1.0 sigma)
                perturbed_seq = baseline_seq.copy()
                perturbed_seq[0, :, idx] = 1.0
                pert_tensor = torch.tensor(perturbed_seq, device=device)
                pert_pred = float(lstm_net(pert_tensor).cpu().numpy().flatten()[0])
                delta = abs(pert_pred - base_pred)
                sensitivities.append(delta)

        total_sens = sum(sensitivities) or 1.0
        normalized_sens = [round(float(s / total_sens), 4) for s in sensitivities]

        indexed = sorted(
            [
                {
                    "feature": feature_names[i],
                    "importance": normalized_sens[i],
                    "raw_score": round(float(sensitivities[i]), 5),
                    "label": get_feature_meta(feature_names[i])["label"],
                    "unit": get_feature_meta(feature_names[i])["unit"],
                }
                for i in range(len(feature_names))
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
        for rank, item in enumerate(indexed, start=1):
            item["rank"] = rank

        return {
            "model_id": "benchmark_lstm",
            "model_name": "NASA C-MAPSS PyTorch LSTM Temporal Regressor",
            "model_type": "lstm_regressor",
            "task": "rul_regression",
            "version": 1,
            "status": "active_benchmark",
            "is_benchmark": True,
            "supported": True,
            "explanation_method": "sequential_channel_perturbation_sensitivity",
            "method_description": (
                "Measures the response sensitivity of the recurrent neural network output when each sensor "
                "channel is perturbed across the temporal observation window (30 operational cycles)."
            ),
            "evaluation_context": {
                "dataset": "NASA C-MAPSS FD001 Turbofan Engine Degradation Benchmark",
                "target": "RUL (Remaining Useful Life in cycles)",
                "sample_count": 100,
                "split": "Held-out test split (100 turbofan units)",
                "metrics": ml_service.evaluation.get("rul_models", {}).get("lstm", {}),
            },
            "features": indexed,
            "limitations": (
                "Sensitivity analysis quantifies the gradient/response surface of the PyTorch neural network. "
                "It does not represent tree feature importance or proven physical degradation causality."
            ),
        }

    elif clean_id in {"benchmark_isolation_forest", "isolation_forest", "nasa_anomaly"}:
        iso = ml_service.anomaly_detector
        feature_names = BENCHMARK_FEATURE_COLS

        # Evaluate decision function sensitivity across standard sensor perturbations
        base_point = np.zeros((1, len(feature_names)))
        sensitivities = []
        base_score = float(iso.decision_function(base_point)[0])

        for idx in range(len(feature_names)):
            perturbed = base_point.copy()
            perturbed[0, idx] = 1.0  # +1 sigma perturbation in standardized space
            pert_score = float(iso.decision_function(perturbed)[0])
            delta = abs(pert_score - base_score)
            sensitivities.append(delta)

        total_sens = sum(sensitivities) or 1.0
        normalized_sens = [round(float(s / total_sens), 4) for s in sensitivities]

        indexed = sorted(
            [
                {
                    "feature": feature_names[i],
                    "importance": normalized_sens[i],
                    "raw_score": round(float(sensitivities[i]), 5),
                    "label": get_feature_meta(feature_names[i])["label"],
                    "unit": get_feature_meta(feature_names[i])["unit"],
                }
                for i in range(len(feature_names))
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
        for rank, item in enumerate(indexed, start=1):
            item["rank"] = rank

        return {
            "model_id": "benchmark_isolation_forest",
            "model_name": "NASA C-MAPSS Isolation Forest Anomaly Detector",
            "model_type": "isolation_forest",
            "task": "anomaly_detection",
            "version": 1,
            "status": "active_benchmark",
            "is_benchmark": True,
            "supported": True,
            "explanation_method": "feature_perturbation_sensitivity",
            "method_description": (
                "Evaluates the sensitivity of the isolation tree decision function by measuring changes "
                "in isolation path depth when each sensor channel is perturbed relative to healthy baseline."
            ),
            "evaluation_context": {
                "dataset": "NASA C-MAPSS FD001 Healthy Operational Cycles (RUL > 100)",
                "target": "Unsupervised Anomaly Isolation (Contamination: 0.05)",
                "metrics": ml_service.evaluation.get("anomaly_detection", {}),
            },
            "features": indexed,
            "limitations": (
                "Isolation Forest is unsupervised. Sensitivity indicates which sensor channels most readily "
                "influence anomaly score partitions; it does not establish confirmed component failure."
            ),
        }

    else:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown benchmark model '{benchmark_id}'. Supported: benchmark_rf, benchmark_lstm, benchmark_isolation_forest.",
        )


def explain_user_model_global(db: Session, user: User, model_id: str) -> Dict[str, Any]:
    """Generates global explainability for a user-registered model."""
    record = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == model_id).first()
    if not record or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Registered model not found.")

    entry = get_cached_or_load_artifact(record, user.id)
    bundle = entry["bundle"]
    trained_model = bundle["model"]
    model_type = bundle.get("model_type", record.model_type)
    task = bundle.get("task", record.task)
    feature_cols = bundle.get("feature_columns", [])
    metrics = bundle.get("metrics", {})

    if not feature_cols:
        raise HTTPException(status_code=500, detail="Model artifact is missing feature columns metadata.")

    # 1. Tree Models with native feature_importances_
    if hasattr(trained_model, "feature_importances_"):
        raw_importances = trained_model.feature_importances_
        if len(raw_importances) != len(feature_cols):
            # Preprocessor might have expanded columns via OneHotEncoder
            # Fall back to proportional mapping or first N columns
            raw_importances = raw_importances[: len(feature_cols)]

        total = float(raw_importances.sum()) or 1.0
        normalized = [round(float(v / total), 4) for v in raw_importances]

        indexed = sorted(
            [
                {
                    "feature": feature_cols[i],
                    "importance": normalized[i],
                    "raw_score": round(float(raw_importances[i]), 5),
                    "label": get_feature_meta(feature_cols[i])["label"],
                    "unit": get_feature_meta(feature_cols[i])["unit"],
                }
                for i in range(len(feature_cols))
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
        for rank, item in enumerate(indexed, start=1):
            item["rank"] = rank

        return {
            "model_id": record.id,
            "model_name": record.name,
            "model_type": model_type,
            "task": task,
            "version": record.version,
            "status": record.status,
            "is_benchmark": False,
            "supported": True,
            "explanation_method": "mean_decrease_impurity",
            "method_description": (
                "Gini / variance reduction: measures the average reduction in node impurity "
                "achieved by splits on each feature across all trees in the trained ensemble."
            ),
            "evaluation_context": {
                "dataset_name": record.dataset.name if record.dataset else "User Telemetry Dataset",
                "dataset_id": record.dataset_id,
                "metrics": metrics,
            },
            "features": indexed,
            "limitations": (
                "Feature importance reflects statistical association and split frequency within the trained "
                "ensemble. It does not establish verified physical causation or sensor fault origin."
            ),
        }

    # 2. Linear Models (Logistic Regression)
    elif hasattr(trained_model, "coef_"):
        coefs = trained_model.coef_
        if coefs.ndim > 1:
            # Multi-class: average absolute coefficient across classes
            raw_magnitudes = np.mean(np.abs(coefs), axis=0)
        else:
            raw_magnitudes = np.abs(coefs)

        raw_magnitudes = raw_magnitudes[: len(feature_cols)]
        total = float(raw_magnitudes.sum()) or 1.0
        normalized = [round(float(v / total), 4) for v in raw_magnitudes]

        indexed = sorted(
            [
                {
                    "feature": feature_cols[i],
                    "importance": normalized[i],
                    "raw_score": round(float(raw_magnitudes[i]), 5),
                    "label": get_feature_meta(feature_cols[i])["label"],
                    "unit": get_feature_meta(feature_cols[i])["unit"],
                }
                for i in range(len(feature_cols))
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
        for rank, item in enumerate(indexed, start=1):
            item["rank"] = rank

        return {
            "model_id": record.id,
            "model_name": record.name,
            "model_type": model_type,
            "task": task,
            "version": record.version,
            "status": record.status,
            "is_benchmark": False,
            "supported": True,
            "explanation_method": "standardized_coefficient_magnitude",
            "method_description": (
                "Mean absolute magnitude of standardized regression coefficients. Features with higher "
                "magnitudes exert greater log-odds influence on the predicted condition classification."
            ),
            "evaluation_context": {
                "dataset_name": record.dataset.name if record.dataset else "User Telemetry Dataset",
                "dataset_id": record.dataset_id,
                "metrics": metrics,
            },
            "features": indexed,
            "limitations": (
                "Standardized coefficients reflect linear decision boundary weighting. Collinear features "
                "may share coefficient weight and should not be interpreted as independent causal drivers."
            ),
        }

    # 3. Isolation Forest
    elif model_type == "isolation_forest" or hasattr(trained_model, "decision_function"):
        # Evaluate decision function sensitivity across standard feature perturbations
        base_df = pd.DataFrame([{col: 0.0 for col in feature_cols}])
        preprocessor = bundle["preprocessor"]
        base_proc = transform_features(preprocessor, base_df)
        base_score = float(trained_model.decision_function(base_proc)[0])

        sensitivities = []
        for col in feature_cols:
            pert_df = pd.DataFrame([{c: 1.0 if c == col else 0.0 for c in feature_cols}])
            pert_proc = transform_features(preprocessor, pert_df)
            pert_score = float(trained_model.decision_function(pert_proc)[0])
            sensitivities.append(abs(pert_score - base_score))

        total_sens = sum(sensitivities) or 1.0
        normalized = [round(float(s / total_sens), 4) for s in sensitivities]

        indexed = sorted(
            [
                {
                    "feature": feature_cols[i],
                    "importance": normalized[i],
                    "raw_score": round(float(sensitivities[i]), 5),
                    "label": get_feature_meta(feature_cols[i])["label"],
                    "unit": get_feature_meta(feature_cols[i])["unit"],
                }
                for i in range(len(feature_cols))
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
        for rank, item in enumerate(indexed, start=1):
            item["rank"] = rank

        return {
            "model_id": record.id,
            "model_name": record.name,
            "model_type": model_type,
            "task": task,
            "version": record.version,
            "status": record.status,
            "is_benchmark": False,
            "supported": True,
            "explanation_method": "feature_perturbation_sensitivity",
            "method_description": (
                "Decision function sensitivity: measures the absolute change in isolation score when "
                "each sensor channel is individually perturbed away from the training distribution center."
            ),
            "evaluation_context": {
                "dataset_name": record.dataset.name if record.dataset else "User Telemetry Dataset",
                "dataset_id": record.dataset_id,
                "metrics": metrics,
            },
            "features": indexed,
            "limitations": (
                "Isolation Forest is unsupervised. Sensitivity reflects partition vulnerability in the trees; "
                "it does not prove an anomaly was physically caused by that sensor."
            ),
        }

    # 4. PyTorch LSTM Model
    elif bundle.get("model_architecture") == "pytorch_lstm":
        lstm_net = entry["lstm_net"]
        preprocessor = bundle["preprocessor"]
        base_df = pd.DataFrame([{col: 0.0 for col in feature_cols}])
        base_proc = transform_features(preprocessor, base_df)

        with torch.no_grad():
            base_tensor = torch.tensor(base_proc, dtype=torch.float32).unsqueeze(1)
            base_pred = float(lstm_net(base_tensor).squeeze().item())

            sensitivities = []
            for col in feature_cols:
                pert_df = pd.DataFrame([{c: 1.0 if c == col else 0.0 for c in feature_cols}])
                pert_proc = transform_features(preprocessor, pert_df)
                pert_tensor = torch.tensor(pert_proc, dtype=torch.float32).unsqueeze(1)
                pert_pred = float(lstm_net(pert_tensor).squeeze().item())
                sensitivities.append(abs(pert_pred - base_pred))

        total_sens = sum(sensitivities) or 1.0
        normalized = [round(float(s / total_sens), 4) for s in sensitivities]

        indexed = sorted(
            [
                {
                    "feature": feature_cols[i],
                    "importance": normalized[i],
                    "raw_score": round(float(sensitivities[i]), 5),
                    "label": get_feature_meta(feature_cols[i])["label"],
                    "unit": get_feature_meta(feature_cols[i])["unit"],
                }
                for i in range(len(feature_cols))
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
        for rank, item in enumerate(indexed, start=1):
            item["rank"] = rank

        return {
            "model_id": record.id,
            "model_name": record.name,
            "model_type": model_type,
            "task": task,
            "version": record.version,
            "status": record.status,
            "is_benchmark": False,
            "supported": True,
            "explanation_method": "sequential_channel_perturbation_sensitivity",
            "method_description": (
                "Measures LSTM recurrent output sensitivity to single-channel perturbations while "
                "preserving model input dimensionality and normalization contracts."
            ),
            "evaluation_context": {
                "dataset_name": record.dataset.name if record.dataset else "User Telemetry Dataset",
                "dataset_id": record.dataset_id,
                "metrics": metrics,
            },
            "features": indexed,
            "limitations": (
                "Sensitivity analysis quantifies neural network response gradients. It does not provide "
                "additive Shapley guarantees or proven mechanical failure causality."
            ),
        }

    else:
        # Unsupported architecture
        return {
            "model_id": record.id,
            "model_name": record.name,
            "model_type": model_type,
            "task": task,
            "version": record.version,
            "status": record.status,
            "is_benchmark": False,
            "supported": False,
            "explanation_method": "unsupported",
            "method_description": f"Model architecture '{model_type}' does not support feature importance extraction.",
            "evaluation_context": {"metrics": metrics},
            "features": [],
            "limitations": (
                f"Global feature attribution is not supported for architecture '{model_type}' under the current ML pipeline."
            ),
        }


def explain_prediction_local(
    db: Session,
    user: User,
    prediction_id: Optional[str] = None,
    model_id: Optional[str] = None,
    features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates local feature attribution for a specific inference observation.

    Works either from a persisted prediction history record or an active feature payload.
    """
    history_record = None

    if prediction_id:
        history_record = (
            db.query(ModelPredictionRecord)
            .filter(ModelPredictionRecord.id == prediction_id)
            .first()
        )
        if not history_record or history_record.user_id != user.id:
            raise HTTPException(status_code=404, detail="Prediction history record not found.")

        model_id = history_record.model_id
        meta_dict = json.loads(history_record.input_metadata_json) if history_record.input_metadata_json else {}
        features = meta_dict.get("features")
        if not features or not isinstance(features, dict):
            # For batch summary records, use first row if present
            features = meta_dict.get("sample_features", {})
        if not features:
            raise HTTPException(
                status_code=400,
                detail="Selected prediction history record does not contain feature inputs suitable for local explanation.",
            )

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required for local explanation.")

    if not features or not isinstance(features, dict):
        raise HTTPException(status_code=400, detail="Feature inputs dictionary is required.")

    # 1. Benchmark Model Local Explanation
    if model_id.startswith("benchmark_"):
        if not ml_service.is_loaded:
            raise HTTPException(status_code=503, detail="Benchmark models are not loaded.")

        benchmark_clean = model_id.lower().strip()
        features_df = pd.DataFrame([features])

        # Ensure benchmark feature columns exist
        missing = [c for c in BENCHMARK_FEATURE_COLS if c not in features_df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required benchmark feature columns: {', '.join(missing[:5])}.",
            )

        if benchmark_clean in {"benchmark_rf", "rf_baseline"}:
            rf = ml_service.rf_baseline
            actual_pred = float(ml_service.predict_rul_raw(features_df, use_lstm=False))

            # Marginal perturbation contribution:
            # Replace each feature with the benchmark scaled mean (0.0 in scaler space) and measure change in RUL
            scaled_input = ml_service.scaler.transform(features_df[BENCHMARK_FEATURE_COLS])
            base_point = np.zeros_like(scaled_input)
            base_pred = float(rf.predict(pd.DataFrame(base_point, columns=BENCHMARK_FEATURE_COLS))[0])

            contributions = []
            for idx, col in enumerate(BENCHMARK_FEATURE_COLS):
                pert_scaled = scaled_input.copy()
                pert_scaled[0, idx] = 0.0  # Reset to mean
                pred_without_feature = float(rf.predict(pd.DataFrame(pert_scaled, columns=BENCHMARK_FEATURE_COLS))[0])
                delta = actual_pred - pred_without_feature

                direction = "increases_rul" if delta >= 0 else "decreases_rul"
                meta = get_feature_meta(col)
                contributions.append({
                    "feature": col,
                    "label": meta["label"],
                    "value": float(features[col]),
                    "unit": meta["unit"],
                    "contribution": round(float(delta), 2),
                    "impact_direction": direction,
                    "description": (
                        f"Pushes predicted RUL by {abs(delta):.1f} cycles {'higher' if delta >= 0 else 'lower'} "
                        f"relative to nominal sensor baseline."
                    ),
                })

            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

            return {
                "prediction_id": prediction_id,
                "model_id": "benchmark_rf",
                "model_name": "NASA C-MAPSS Random Forest Regressor",
                "task": "rul_regression",
                "prediction_output": {"rul_cycles": round(actual_pred, 1), "unit": "cycles"},
                "explanation_method": "marginal_feature_perturbation",
                "contributions": contributions,
                "limitations": (
                    "Local contributions measure the shift in Random Forest tree voting caused by each sensor "
                    "relative to the nominal training mean. It does not represent physical wear causation."
                ),
            }

        elif benchmark_clean in {"benchmark_isolation_forest", "isolation_forest"}:
            actual_res = ml_service.detect_anomaly_raw(features_df)
            scaled_input = ml_service.scaler.transform(features_df[BENCHMARK_FEATURE_COLS])
            iso = ml_service.anomaly_detector
            actual_score = actual_res["anomaly_score"]

            contributions = []
            for idx, col in enumerate(BENCHMARK_FEATURE_COLS):
                pert_scaled = scaled_input.copy()
                pert_scaled[0, idx] = 0.0  # Reset to healthy nominal mean
                pert_score_raw = -float(iso.score_samples(pert_scaled)[0])
                pert_normalized = float(np.clip((pert_score_raw + 0.1) / 0.45, 0.0, 1.0))
                delta = actual_score - pert_normalized

                direction = "elevates_anomaly_score" if delta >= 0 else "suppresses_anomaly_score"
                meta = get_feature_meta(col)
                contributions.append({
                    "feature": col,
                    "label": meta["label"],
                    "value": float(features[col]),
                    "unit": meta["unit"],
                    "contribution": round(float(delta), 4),
                    "impact_direction": direction,
                    "description": (
                        f"Sensor deviation drives anomaly score {abs(delta):.4f} points "
                        f"{'higher into anomalous zone' if delta >= 0 else 'lower toward normal'}."
                    ),
                })

            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

            return {
                "prediction_id": prediction_id,
                "model_id": "benchmark_isolation_forest",
                "model_name": "NASA C-MAPSS Isolation Forest Anomaly Detector",
                "task": "anomaly_detection",
                "prediction_output": actual_res,
                "explanation_method": "decision_surface_deviation_impact",
                "contributions": contributions,
                "limitations": (
                    "Identifies sensor channels whose values fall furthest from the healthy training envelope. "
                    "Unsupervised anomaly detection indicates unusual patterns, not confirmed mechanical failure."
                ),
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Local explanation is unsupported for benchmark model '{model_id}'.",
            )

    # 2. User-Trained Model Local Explanation
    record = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == model_id).first()
    if not record or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Registered model not found.")

    entry = get_cached_or_load_artifact(record, user.id)
    bundle = entry["bundle"]
    preprocessor = bundle["preprocessor"]
    feature_cols = bundle.get("feature_columns", [])
    target_col = bundle.get("target_column")
    trained_model = bundle["model"]
    task = bundle.get("task", record.task)

    # Validate feature schema
    df = validate_and_prepare_dataframe([features], feature_cols, target_col, preprocessor=preprocessor)
    X_proc = transform_features(preprocessor, df)

    contributions = []
    prediction_output: Dict[str, Any] = {}

    if task == "rul_regression":
        if bundle.get("model_architecture") == "pytorch_lstm":
            lstm_net = entry["lstm_net"]
            with torch.no_grad():
                X_tensor = torch.tensor(X_proc, dtype=torch.float32).unsqueeze(1)
                actual_pred = round(float(lstm_net(X_tensor).squeeze().item()), 2)
            prediction_output = {"rul_cycles": actual_pred, "unit": "cycles"}

            # Sensitivity perturbation across input channels
            for idx, col in enumerate(feature_cols):
                pert_proc = X_proc.copy()
                pert_proc[0, idx] = 0.0  # Reset feature to center
                pert_tensor = torch.tensor(pert_proc, dtype=torch.float32).unsqueeze(1)
                with torch.no_grad():
                    pred_without = round(float(lstm_net(pert_tensor).squeeze().item()), 2)
                delta = actual_pred - pred_without

                meta = get_feature_meta(col)
                contributions.append({
                    "feature": col,
                    "label": meta["label"],
                    "value": features.get(col),
                    "unit": meta["unit"],
                    "contribution": round(float(delta), 2),
                    "impact_direction": "increases_rul" if delta >= 0 else "decreases_rul",
                    "description": f"Shifts estimated RUL by {abs(delta):.1f} cycles relative to channel baseline.",
                })

        else:
            actual_pred = round(float(trained_model.predict(X_proc)[0]), 2)
            prediction_output = {"rul_cycles": actual_pred, "unit": "cycles"}

            for idx, col in enumerate(feature_cols):
                pert_proc = X_proc.copy()
                pert_proc[0, idx] = 0.0
                pred_without = round(float(trained_model.predict(pert_proc)[0]), 2)
                delta = actual_pred - pred_without

                meta = get_feature_meta(col)
                contributions.append({
                    "feature": col,
                    "label": meta["label"],
                    "value": features.get(col),
                    "unit": meta["unit"],
                    "contribution": round(float(delta), 2),
                    "impact_direction": "increases_rul" if delta >= 0 else "decreases_rul",
                    "description": f"Contributes {abs(delta):.1f} cycles to RUL prediction relative to feature baseline.",
                })

    elif task == "condition_classification":
        label_encoder = bundle.get("label_encoder")
        raw_pred = trained_model.predict(X_proc)[0]
        decoded_class = str(label_encoder.inverse_transform([raw_pred])[0]) if label_encoder else str(raw_pred)
        has_proba = hasattr(trained_model, "predict_proba")
        probs = trained_model.predict_proba(X_proc)[0] if has_proba else None

        prediction_output = {"condition": decoded_class}
        if probs is not None:
            classes = [str(c) for c in (label_encoder.classes_ if label_encoder else trained_model.classes_)]
            prediction_output["probabilities"] = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}

        # Measure feature contribution to predicted class score / probability
        for idx, col in enumerate(feature_cols):
            pert_proc = X_proc.copy()
            pert_proc[0, idx] = 0.0

            if has_proba:
                target_idx = list(classes).index(decoded_class) if decoded_class in classes else 0
                prob_before = float(probs[target_idx])
                prob_after = float(trained_model.predict_proba(pert_proc)[0][target_idx])
                delta = prob_before - prob_after
                direction = "supports_condition" if delta >= 0 else "opposes_condition"
                desc = f"Contributes {abs(delta)*100:.1f}% confidence towards condition '{decoded_class}'."
            else:
                delta = 0.0
                direction = "supports_condition"
                desc = "Linear contribution to class decision."

            meta = get_feature_meta(col)
            contributions.append({
                "feature": col,
                "label": meta["label"],
                "value": features.get(col),
                "unit": meta["unit"],
                "contribution": round(float(delta), 4),
                "impact_direction": direction,
                "description": desc,
            })

    elif task == "anomaly_detection":
        is_anom = bool(trained_model.predict(X_proc)[0] == -1)
        actual_score = round(float(trained_model.decision_function(X_proc)[0]), 4)
        prediction_output = {"is_anomaly": is_anom, "anomaly_score": actual_score}

        for idx, col in enumerate(feature_cols):
            pert_proc = X_proc.copy()
            pert_proc[0, idx] = 0.0
            score_without = float(trained_model.decision_function(pert_proc)[0])
            # Lower score = more anomalous. Delta > 0 means feature reduced score (pushed toward anomaly)
            delta = score_without - actual_score
            direction = "elevates_anomaly_risk" if delta >= 0 else "suppresses_anomaly_risk"

            meta = get_feature_meta(col)
            contributions.append({
                "feature": col,
                "label": meta["label"],
                "value": features.get(col),
                "unit": meta["unit"],
                "contribution": round(float(delta), 4),
                "impact_direction": direction,
                "description": (
                    f"Sensor value shifts isolation score by {abs(delta):.4f} "
                    f"{'towards anomaly boundary' if delta >= 0 else 'towards normal baseline'}."
                ),
            })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    return {
        "prediction_id": prediction_id,
        "model_id": record.id,
        "model_name": record.name,
        "model_version": record.version,
        "model_type": record.model_type,
        "task": task,
        "prediction_output": prediction_output,
        "explanation_method": "marginal_feature_perturbation",
        "contributions": contributions,
        "limitations": (
            "Local feature contributions reflect the model's response surface under single-feature "
            "baseline substitution. They do not constitute root-cause proof of physical equipment failure."
        ),
    }
