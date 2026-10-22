"""Inference Service for MachinaSense Backend Part 4.

Provides real, authenticated inference for active user-trained ML models:
- RUL Regression (Random Forest, Gradient Boosting, PyTorch LSTM)
- Condition Classification (Random Forest, Gradient Boosting, Logistic Regression)
- Anomaly Detection (Isolation Forest)
- Batch Inference
- Prediction Persistence & Audit History
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ModelPredictionRecord, ModelRegistryRecord, User
from app.services.training_engine import PyTorchLSTMRUL

logger = logging.getLogger("machinasense.inference")

SUPPORTED_TASKS = {"rul_regression", "condition_classification", "anomaly_detection"}

# In-process cache: key = (model_id, version, user_id) -> (bundle, optional_lstm_net)
_model_cache: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
_cache_lock = threading.Lock()


def get_cached_or_load_artifact(model: ModelRegistryRecord, user_id: str) -> Dict[str, Any]:
    """Loads and caches the model artifact bundle safely in memory."""
    cache_key = (model.id, model.version, user_id)
    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

    # Verify path safety
    artifact_path = model.artifact_path
    if not artifact_path or not isinstance(artifact_path, str):
        raise HTTPException(status_code=500, detail="Active model artifact path is invalid.")

    if ".." in artifact_path:
        raise HTTPException(status_code=400, detail="Path traversal in artifact path is not permitted.")

    norm_path = os.path.abspath(artifact_path)
    base_dir = os.path.abspath(settings.MODELS_DIR)
    try:
        common = os.path.commonpath([norm_path, base_dir])
    except ValueError:
        raise HTTPException(status_code=400, detail="Artifact path is outside authorized storage.")

    if common != base_dir:
        raise HTTPException(status_code=400, detail="Artifact path is outside authorized storage.")

    if not os.path.exists(norm_path):
        raise HTTPException(status_code=500, detail="Active model artifact file is missing from storage.")

    if not os.path.isfile(norm_path) or not os.access(norm_path, os.R_OK):
        raise HTTPException(status_code=500, detail="Active model artifact file is unreadable.")

    try:
        bundle = joblib.load(norm_path)
    except Exception as exc:
        logger.exception("Failed to load model artifact %s: %s", norm_path, exc)
        raise HTTPException(status_code=500, detail="Model artifact file is corrupt or unreadable.") from exc

    if not isinstance(bundle, dict) or "model" not in bundle or "preprocessor" not in bundle:
        raise HTTPException(status_code=500, detail="Model artifact structure is malformed.")

    lstm_net = None
    if bundle.get("model_architecture") == "pytorch_lstm":
        state_dict = bundle["model"]
        # Determine input dimension from saved weights
        input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
        lstm_net = PyTorchLSTMRUL(input_dim=input_dim, hidden_dim=32, num_layers=1)
        lstm_net.load_state_dict(state_dict)
        lstm_net.eval()

    entry = {
        "bundle": bundle,
        "lstm_net": lstm_net,
        "loaded_at": datetime.now(timezone.utc),
    }

    with _cache_lock:
        _model_cache[cache_key] = entry

    return entry


def clear_model_cache() -> None:
    """Clears the in-memory model cache."""
    with _cache_lock:
        _model_cache.clear()


def resolve_active_model(db: Session, user: User, task: str) -> ModelRegistryRecord:
    """Resolves and validates the active model for the given task and user."""
    if task not in SUPPORTED_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Supported tasks: {', '.join(sorted(SUPPORTED_TASKS))}.",
        )

    model = (
        db.query(ModelRegistryRecord)
        .filter(
            ModelRegistryRecord.user_id == user.id,
            ModelRegistryRecord.task == task,
            ModelRegistryRecord.status == "active",
        )
        .first()
    )

    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"No active model found for task '{task}'.",
        )

    return model


def validate_and_prepare_dataframe(
    raw_rows: List[Dict[str, Any]],
    feature_cols: List[str],
    target_col: Optional[str],
    preprocessor: Any = None,
) -> pd.DataFrame:
    """Validates incoming feature rows against training feature schema and prepares a DataFrame."""
    if not raw_rows:
        raise HTTPException(status_code=400, detail="No feature data provided for inference.")

    for i, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise HTTPException(status_code=400, detail=f"Row {i} must be a dictionary of features.")

        # 1. Reject target column in features
        if target_col and target_col in row:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_col}' cannot be provided as an input feature.",
            )

        # 2. Reject missing required features
        missing = [col for col in feature_cols if col not in row]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature columns: {', '.join(missing)}.",
            )

    df = pd.DataFrame(raw_rows)

    # 3. Restrict DataFrame to required feature columns
    df = df[feature_cols].copy()

    # Determine numeric columns from preprocessor if available
    num_cols = set()
    if preprocessor and hasattr(preprocessor, "transformers"):
        for trans in preprocessor.transformers:
            if len(trans) >= 3 and trans[0] == "num":
                num_cols.update(trans[2])

    # 4. Check that numeric features are valid/convertible to numeric
    for col in feature_cols:
        val_sample = df[col].dropna()
        if not val_sample.empty:
            first_val = val_sample.iloc[0]
            is_num_col = col in num_cols or isinstance(first_val, (int, float, np.number))
            if is_num_col:
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid non-numeric value provided for feature column '{col}'.",
                    )
            elif isinstance(first_val, str):
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                except Exception:
                    pass

    return df


def transform_features(preprocessor: Any, df: pd.DataFrame) -> np.ndarray:
    """Transforms DataFrame using fitted preprocessor with safe error handling."""
    try:
        return preprocessor.transform(df)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature values for model transformation: {str(exc)}",
        )


def execute_rul_inference(
    db: Session,
    user: User,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs RUL regression inference using the user's active RUL model."""
    if not isinstance(features, dict) or not features:
        raise HTTPException(status_code=400, detail="Features dictionary is required and cannot be empty.")
    if len(features) > 200:
        raise HTTPException(status_code=400, detail="Features dictionary exceeds maximum allowed limit of 200 items.")

    model = resolve_active_model(db, user, "rul_regression")
    entry = get_cached_or_load_artifact(model, user.id)
    bundle = entry["bundle"]
    preprocessor = bundle["preprocessor"]
    feature_cols = bundle["feature_columns"]
    target_col = bundle.get("target_column")

    df = validate_and_prepare_dataframe([features], feature_cols, target_col, preprocessor=preprocessor)
    X_proc = transform_features(preprocessor, df)

    if bundle.get("model_architecture") == "pytorch_lstm":
        lstm_net = entry["lstm_net"]
        with torch.no_grad():
            X_tensor = torch.tensor(X_proc, dtype=torch.float32).unsqueeze(1)
            pred = float(lstm_net(X_tensor).squeeze().item())
    else:
        trained_model = bundle["model"]
        pred = float(trained_model.predict(X_proc)[0])

    rul_cycles = round(pred, 2)
    prediction_result = {"rul_cycles": rul_cycles}

    # Persist prediction
    persist_prediction_record(
        db=db,
        user_id=user.id,
        model_id=model.id,
        task="rul_regression",
        input_metadata={"type": "single", "features": features},
        prediction=prediction_result,
    )

    return {
        "task": "rul_regression",
        "prediction": prediction_result,
        "model": {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "model_type": model.model_type,
        },
    }


def execute_classification_inference(
    db: Session,
    user: User,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs condition classification inference using the user's active classification model."""
    if not isinstance(features, dict) or not features:
        raise HTTPException(status_code=400, detail="Features dictionary is required and cannot be empty.")
    if len(features) > 200:
        raise HTTPException(status_code=400, detail="Features dictionary exceeds maximum allowed limit of 200 items.")

    model = resolve_active_model(db, user, "condition_classification")
    entry = get_cached_or_load_artifact(model, user.id)
    bundle = entry["bundle"]
    preprocessor = bundle["preprocessor"]
    label_encoder = bundle.get("label_encoder")
    trained_model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    target_col = bundle.get("target_column")

    df = validate_and_prepare_dataframe([features], feature_cols, target_col, preprocessor=preprocessor)
    X_proc = transform_features(preprocessor, df)

    encoded_pred = trained_model.predict(X_proc)
    if label_encoder is not None:
        decoded_class = str(label_encoder.inverse_transform(encoded_pred)[0])
        class_names = [str(c) for c in label_encoder.classes_]
    else:
        decoded_class = str(encoded_pred[0])
        class_names = [str(c) for c in trained_model.classes_]

    prediction_result: Dict[str, Any] = {"condition": decoded_class}

    # Add probabilities if supported
    if hasattr(trained_model, "predict_proba"):
        try:
            probs = trained_model.predict_proba(X_proc)[0]
            prediction_result["probabilities"] = {
                class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))
            }
        except Exception:
            pass

    # Persist prediction
    persist_prediction_record(
        db=db,
        user_id=user.id,
        model_id=model.id,
        task="condition_classification",
        input_metadata={"type": "single", "features": features},
        prediction=prediction_result,
    )

    return {
        "task": "condition_classification",
        "prediction": prediction_result,
        "model": {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "model_type": model.model_type,
        },
    }


def execute_anomaly_inference(
    db: Session,
    user: User,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs anomaly detection inference using the user's active Isolation Forest model."""
    if not isinstance(features, dict) or not features:
        raise HTTPException(status_code=400, detail="Features dictionary is required and cannot be empty.")
    if len(features) > 200:
        raise HTTPException(status_code=400, detail="Features dictionary exceeds maximum allowed limit of 200 items.")

    model = resolve_active_model(db, user, "anomaly_detection")
    entry = get_cached_or_load_artifact(model, user.id)
    bundle = entry["bundle"]
    preprocessor = bundle["preprocessor"]
    trained_model = bundle["model"]
    feature_cols = bundle["feature_columns"]

    df = validate_and_prepare_dataframe([features], feature_cols, target_col=None, preprocessor=preprocessor)
    X_proc = transform_features(preprocessor, df)

    # Isolation Forest: -1 = anomaly, 1 = normal
    raw_pred = trained_model.predict(X_proc)[0]
    score = float(trained_model.decision_function(X_proc)[0])

    prediction_result = {
        "is_anomaly": bool(raw_pred == -1),
        "anomaly_score": round(score, 4),
    }

    # Persist prediction
    persist_prediction_record(
        db=db,
        user_id=user.id,
        model_id=model.id,
        task="anomaly_detection",
        input_metadata={"type": "single", "features": features},
        prediction=prediction_result,
    )

    return {
        "task": "anomaly_detection",
        "prediction": prediction_result,
        "model": {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "model_type": model.model_type,
        },
    }


def execute_batch_inference(
    db: Session,
    user: User,
    task: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs batch inference for RUL, classification, or anomaly detection."""
    if task not in SUPPORTED_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Supported tasks: {', '.join(sorted(SUPPORTED_TASKS))}.",
        )

    rows = payload.get("rows")
    if not rows or not isinstance(rows, list) or len(rows) == 0:
        raise HTTPException(
            status_code=400,
            detail="Batch inference request must contain a non-empty 'rows' list.",
        )

    if len(rows) > settings.INFERENCE_MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size ({len(rows)}) exceeds maximum allowed ({settings.INFERENCE_MAX_BATCH_SIZE}).",
        )

    model = resolve_active_model(db, user, task)
    entry = get_cached_or_load_artifact(model, user.id)
    bundle = entry["bundle"]
    preprocessor = bundle["preprocessor"]
    feature_cols = bundle["feature_columns"]
    target_col = bundle.get("target_column")
    trained_model = bundle["model"]

    df = validate_and_prepare_dataframe(rows, feature_cols, target_col, preprocessor=preprocessor)
    X_proc = transform_features(preprocessor, df)

    predictions_list: List[Dict[str, Any]] = []

    if task == "rul_regression":
        if bundle.get("model_architecture") == "pytorch_lstm":
            lstm_net = entry["lstm_net"]
            with torch.no_grad():
                X_tensor = torch.tensor(X_proc, dtype=torch.float32).unsqueeze(1)
                preds = lstm_net(X_tensor).squeeze(-1).numpy().flatten()
        else:
            preds = trained_model.predict(X_proc)

        for p in preds:
            predictions_list.append({"rul_cycles": round(float(p), 2)})

    elif task == "condition_classification":
        label_encoder = bundle.get("label_encoder")
        encoded_preds = trained_model.predict(X_proc)
        if label_encoder is not None:
            decoded_classes = label_encoder.inverse_transform(encoded_preds)
            class_names = [str(c) for c in label_encoder.classes_]
        else:
            decoded_classes = encoded_preds
            class_names = [str(c) for c in trained_model.classes_]

        has_proba = hasattr(trained_model, "predict_proba")
        probs = trained_model.predict_proba(X_proc) if has_proba else None

        for i, c in enumerate(decoded_classes):
            item = {"condition": str(c)}
            if probs is not None:
                item["probabilities"] = {
                    class_names[j]: round(float(probs[i][j]), 4) for j in range(len(class_names))
                }
            predictions_list.append(item)

    elif task == "anomaly_detection":
        preds = trained_model.predict(X_proc)
        scores = trained_model.decision_function(X_proc)
        for i in range(len(preds)):
            predictions_list.append({
                "is_anomaly": bool(preds[i] == -1),
                "anomaly_score": round(float(scores[i]), 4),
            })

    # Persist summary for batch
    persist_prediction_record(
        db=db,
        user_id=user.id,
        model_id=model.id,
        task=task,
        input_metadata={"type": "batch", "batch_size": len(rows), "columns": feature_cols},
        prediction={
            "batch_summary": True,
            "total_predictions": len(predictions_list),
            "sample_first": predictions_list[0] if predictions_list else None,
        },
    )

    return {
        "task": task,
        "predictions": predictions_list,
        "model": {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "model_type": model.model_type,
        },
        "total_rows": len(rows),
    }


def persist_prediction_record(
    db: Session,
    user_id: str,
    model_id: str,
    task: str,
    input_metadata: Dict[str, Any],
    prediction: Dict[str, Any],
) -> ModelPredictionRecord:
    """Persists an auditable inference record to the model_predictions table."""
    record = ModelPredictionRecord(
        id=f"pred_{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        model_id=model_id,
        task=task,
        input_metadata_json=json.dumps(input_metadata),
        prediction_json=json.dumps(prediction),
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_prediction_history(
    db: Session,
    user: User,
    task: Optional[str] = None,
    model_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Retrieves prediction audit history scoped to authenticated user."""
    safe_limit = max(1, min(limit, 100))
    query = db.query(ModelPredictionRecord).filter(ModelPredictionRecord.user_id == user.id)

    if task:
        query = query.filter(ModelPredictionRecord.task == task)
    if model_id:
        query = query.filter(ModelPredictionRecord.model_id == model_id)

    records = query.order_by(ModelPredictionRecord.created_at.desc()).limit(safe_limit).all()

    formatted = [
        {
            "id": r.id,
            "model_id": r.model_id,
            "model_name": r.model.name if r.model else None,
            "model_version": r.model.version if r.model else None,
            "model_type": r.model.model_type if r.model else None,
            "task": r.task,
            "input_metadata": json.loads(r.input_metadata_json) if r.input_metadata_json else {},
            "prediction": json.loads(r.prediction_json) if r.prediction_json else {},
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]

    return {
        "predictions": formatted,
        "total": len(formatted),
    }


def get_prediction_detail(
    db: Session,
    user: User,
    prediction_id: str,
) -> Dict[str, Any]:
    """Retrieves a single prediction audit record, scoped to authenticated user."""
    record = (
        db.query(ModelPredictionRecord)
        .filter(ModelPredictionRecord.id == prediction_id)
        .first()
    )
    if not record or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Prediction record not found.")

    return {
        "id": record.id,
        "model_id": record.model_id,
        "model_name": record.model.name if record.model else None,
        "model_version": record.model.version if record.model else None,
        "model_type": record.model.model_type if record.model else None,
        "task": record.task,
        "input_metadata": json.loads(record.input_metadata_json) if record.input_metadata_json else {},
        "prediction": json.loads(record.prediction_json) if record.prediction_json else {},
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }
