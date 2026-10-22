"""Real ML Training Engine for MachinaSense user-uploaded datasets.
Supports:
- RUL Regression: Random Forest, Gradient Boosting, PyTorch LSTM
- Condition Classification: Random Forest, Gradient Boosting, Logistic Regression
- Anomaly Detection: Isolation Forest
"""
from __future__ import annotations

import datetime
import io
import json
import logging
import os
import shutil
import threading
from typing import Any, Dict, List, Optional, Tuple, Set

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

from app.config import settings
from app.db.models import TrainingJobRecord, UserDataset
from app.db.session import SessionLocal

logger = logging.getLogger("machinasense.training")

SUPPORTED_TASK_MODELS: Dict[str, Set[str]] = {
    "rul_regression": {
        "rf_regressor",
        "gradient_boosting_regressor",
        "lstm_regressor",
    },
    "condition_classification": {
        "rf_classifier",
        "gradient_boosting_classifier",
        "logistic_regression",
    },
    "anomaly_detection": {
        "isolation_forest",
    },
}

_cancelled_jobs: Set[str] = set()
_lock = threading.Lock()


class PyTorchLSTMRUL(nn.Module):
    """PyTorch LSTM for sequential temporal RUL estimation."""
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def request_cancel_job(job_id: str) -> None:
    with _lock:
        _cancelled_jobs.add(job_id)


def is_job_cancelled(job_id: str) -> bool:
    with _lock:
        return job_id in _cancelled_jobs


def validate_training_config(config: Dict[str, Any], dataset: UserDataset, df: pd.DataFrame) -> None:
    """Validates user configuration against dataset schema and algorithm requirements."""
    task = config.get("task")
    if not task or task not in SUPPORTED_TASK_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task '{task}'. Supported tasks: {list(SUPPORTED_TASK_MODELS.keys())}",
        )

    model_type = config.get("model_type")
    valid_models = SUPPORTED_TASK_MODELS[task]
    if not model_type or model_type not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{model_type}' for task '{task}'. Supported models: {list(valid_models)}",
        )

    feature_cols = config.get("feature_columns")
    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise HTTPException(status_code=400, detail="At least one feature column must be specified.")

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Feature columns not found in dataset: {missing_cols}")

    target_col = config.get("target_column")
    if task != "anomaly_detection":
        if not target_col or not isinstance(target_col, str) or not target_col.strip():
            raise HTTPException(status_code=400, detail=f"Target column is required for supervised task '{task}'.")
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"target_column '{target_col}' not found in dataset columns.")
        if target_col in feature_cols:
            raise HTTPException(status_code=400, detail="Target column cannot be included in feature columns.")
    else:
        # Unsupervised task
        if target_col:
            config["target_column"] = None

    # Validate split ratios — accept both fraction (0-1) and percentage (0-100)
    train_split = float(config.get("train_split", 70.0))
    val_split = float(config.get("validation_split", 15.0))
    test_split = float(config.get("test_split", 15.0))

    # Auto-detect: if all splits are <= 1.0, treat as fractions and convert to percentages
    if train_split <= 1.0 and val_split <= 1.0 and test_split <= 1.0:
        train_split *= 100.0
        val_split *= 100.0
        test_split *= 100.0
        config["train_split"] = train_split
        config["validation_split"] = val_split
        config["test_split"] = test_split

    if train_split <= 0 or test_split <= 0 or val_split < 0:
        raise HTTPException(status_code=400, detail="Train and test splits must be greater than 0.")

    total_split = round(train_split + val_split + test_split, 2)
    if total_split != 100.0:
        raise HTTPException(
            status_code=400,
            detail=f"Train/validation/test splits must sum to 1.0 (or 100%). Currently: {total_split}%.",
        )

    seed = config.get("random_seed")
    if seed is None:
        config["random_seed"] = 42
    elif isinstance(seed, bool):
        raise HTTPException(status_code=400, detail="Random seed must be a valid integer.")
    else:
        try:
            config["random_seed"] = int(seed)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Random seed must be a valid integer.")

    if len(df) < 5:
        raise HTTPException(status_code=400, detail="Dataset must contain at least 5 rows for training and evaluation.")


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    """Builds an isolated preprocessing transformer for numeric and categorical columns."""
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        num_pipeline = ColumnTransformer(
            transformers=[
                ("imputer", SimpleImputer(strategy="median"), [0]),
                ("scaler", StandardScaler(), [0]),
            ],
            remainder="drop",
        )
        # Use simpler scikit pipelines for numeric & categorical
        transformers.append((
            "num",
            StandardScaler(),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def _update_job_step(db, job_id: str, status: str, step: str, progress: Optional[float] = None, log: Optional[str] = None):
    job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
    if not job:
        return
    job.status = status
    job.current_step = step
    if progress is not None:
        job.progress = progress
    if log:
        current_logs = json.loads(job.logs_json) if job.logs_json else []
        current_logs.append(f"[{datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}] {log}")
        job.logs_json = json.dumps(current_logs)
    db.commit()


def _mark_job_cancelled(job: TrainingJobRecord, db: Any) -> None:
    job.status = "cancelled"
    job.current_step = "Job cancelled by user"
    job.completed_at = datetime.datetime.now(datetime.timezone.utc)
    current_logs = json.loads(job.logs_json) if job.logs_json else []
    current_logs.append(f"[{datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}] Job cancelled by user.")
    job.logs_json = json.dumps(current_logs)
    db.commit()


def run_training_pipeline(job_id: str, user_id: str) -> None:
    """
    Background worker function executing end-to-end ML training on real dataset.
    Updates database progress sequentially.
    """
    db = SessionLocal()
    try:
        job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
        if not job:
            return

        dataset = db.query(UserDataset).filter(UserDataset.id == job.dataset_id, UserDataset.user_id == user_id).first()
        if not dataset or not os.path.exists(dataset.file_path):
            job.status = "failed"
            job.error_message = "Dataset file could not be located in storage."
            db.commit()
            return

        if is_job_cancelled(job_id):
            _mark_job_cancelled(job, db)
            return

        # 1. Validating
        job.started_at = datetime.datetime.now(datetime.timezone.utc)
        _update_job_step(db, job_id, "validating", "Validating dataset schema and split parameters", progress=15.0, log="Validating configuration against dataset schema...")

        try:
            df = pd.read_csv(dataset.file_path)
        except Exception as exc:
            job.status = "failed"
            job.error_message = f"Failed to load dataset: {str(exc)}"
            db.commit()
            return

        feature_cols: List[str] = json.loads(job.feature_columns_json)
        target_col: Optional[str] = job.target_column
        task: str = job.task
        model_type: str = job.model_type
        seed: int = job.random_seed or 42

        # 2. Preprocessing
        if is_job_cancelled(job_id):
            _mark_job_cancelled(job, db)
            return

        _update_job_step(db, job_id, "preprocessing", "Imputing, encoding, and scaling features", progress=35.0, log="Fitting preprocessor and splitting partitions...")

        # Drop rows where target is NaN for supervised tasks
        if task != "anomaly_detection" and target_col:
            df = df.dropna(subset=[target_col]).copy()

        # Handle missing values in numeric features with median, categorical with mode
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "missing")

        X = df[feature_cols].copy()
        y = df[target_col].copy() if (task != "anomaly_detection" and target_col) else None

        label_encoder: Optional[LabelEncoder] = None
        if task == "condition_classification" and y is not None:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.astype(str))

        # Split data
        test_ratio = job.test_split / 100.0
        val_ratio = job.validation_split / 100.0
        train_ratio = job.train_split / 100.0

        # Initial train+val vs test split
        stratify_y = y if (task == "condition_classification" and y is not None and len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2) else None
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y if y is not None else np.zeros(len(X)),
            test_size=test_ratio,
            random_state=seed,
            stratify=stratify_y,
        )

        # Build and fit preprocessor ONLY on training set
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        transformers = []
        if numeric_cols:
            transformers.append(("num", StandardScaler(), numeric_cols))
        if categorical_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        X_train_val_proc = preprocessor.fit_transform(X_train_val)
        X_test_proc = preprocessor.transform(X_test)

        # 3. Model Training
        if is_job_cancelled(job_id):
            _mark_job_cancelled(job, db)
            return

        _update_job_step(db, job_id, "training", f"Training algorithm: {model_type}", progress=55.0, log=f"Training {model_type} on {len(X_train_val)} samples...")

        trained_model: Any = None

        if task == "rul_regression":
            y_train_val = y_train_val.astype(float)
            y_test = y_test.astype(float)

            if model_type == "rf_regressor":
                rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
                rf.fit(X_train_val_proc, y_train_val)
                trained_model = rf
            elif model_type == "gradient_boosting_regressor":
                gb = GradientBoostingRegressor(n_estimators=50, random_state=seed)
                gb.fit(X_train_val_proc, y_train_val)
                trained_model = gb
            elif model_type == "lstm_regressor":
                # PyTorch LSTM
                torch.manual_seed(seed)
                input_dim = X_train_val_proc.shape[1]
                lstm_net = PyTorchLSTMRUL(input_dim=input_dim, hidden_dim=32, num_layers=1)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(lstm_net.parameters(), lr=0.01)

                # Reshape to (batch, seq_len=1, features)
                X_tensor = torch.tensor(X_train_val_proc, dtype=torch.float32).unsqueeze(1)
                y_tensor = torch.tensor(y_train_val.values if hasattr(y_train_val, 'values') else y_train_val, dtype=torch.float32).unsqueeze(1)

                lstm_net.train()
                for epoch in range(20):
                    optimizer.zero_grad()
                    out = lstm_net(X_tensor)
                    loss = criterion(out, y_tensor)
                    loss.backward()
                    optimizer.step()

                lstm_net.eval()
                trained_model = lstm_net

        elif task == "condition_classification":
            if model_type == "rf_classifier":
                rfc = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
                rfc.fit(X_train_val_proc, y_train_val)
                trained_model = rfc
            elif model_type == "gradient_boosting_classifier":
                gbc = GradientBoostingClassifier(n_estimators=50, random_state=seed)
                gbc.fit(X_train_val_proc, y_train_val)
                trained_model = gbc
            elif model_type == "logistic_regression":
                lr = LogisticRegression(max_iter=500, random_state=seed)
                lr.fit(X_train_val_proc, y_train_val)
                trained_model = lr

        elif task == "anomaly_detection":
            if model_type == "isolation_forest":
                iso = IsolationForest(random_state=seed, contamination=0.05, n_jobs=-1)
                iso.fit(X_train_val_proc)
                trained_model = iso

        # 4. Evaluating
        if is_job_cancelled(job_id):
            _mark_job_cancelled(job, db)
            return

        _update_job_step(db, job_id, "evaluating", "Computing real test evaluation metrics", progress=80.0, log="Evaluating trained model on independent test partition...")

        metrics: Dict[str, Any] = {}

        if task == "rul_regression":
            if model_type == "lstm_regressor":
                with torch.no_grad():
                    X_test_tensor = torch.tensor(X_test_proc, dtype=torch.float32).unsqueeze(1)
                    y_pred = trained_model(X_test_tensor).squeeze(1).numpy()
            else:
                y_pred = trained_model.predict(X_test_proc)

            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 0.0

            metrics = {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "r2": round(r2, 3),
                "test_samples": int(len(y_test)),
            }

        elif task == "condition_classification":
            y_pred = trained_model.predict(X_test_proc)
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            cm = confusion_matrix(y_test, y_pred).tolist()

            classes_list = label_encoder.classes_.tolist() if label_encoder else []

            metrics = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "confusion_matrix": cm,
                "classes": classes_list,
                "test_samples": int(len(y_test)),
            }

        elif task == "anomaly_detection":
            y_pred = trained_model.predict(X_test_proc)
            scores = trained_model.decision_function(X_test_proc)

            anomaly_count = int((y_pred == -1).sum())
            total_samples = int(len(y_pred))
            anomaly_rate = round(float(anomaly_count / total_samples), 4) if total_samples > 0 else 0.0

            metrics = {
                "total_samples": total_samples,
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_rate,
                "contamination": 0.05,
                "score_mean": round(float(scores.mean()), 4),
                "score_min": round(float(scores.min()), 4),
                "score_max": round(float(scores.max()), 4),
                "score_std": round(float(scores.std()), 4),
            }

        # 5. Saving Artifact
        if is_job_cancelled(job_id):
            _mark_job_cancelled(job, db)
            return

        _update_job_step(db, job_id, "saving", "Persisting model artifact and preprocessing pipeline", progress=95.0, log="Saving versioned model artifact to storage...")

        user_models_dir = os.path.join(settings.MODELS_DIR, user_id, job_id)
        os.makedirs(user_models_dir, exist_ok=True)
        artifact_file = os.path.join(user_models_dir, "model_artifact.joblib")

        artifact_bundle = {
            "model": trained_model if model_type != "lstm_regressor" else trained_model.state_dict(),
            "model_architecture": "pytorch_lstm" if model_type == "lstm_regressor" else "sklearn",
            "preprocessor": preprocessor,
            "label_encoder": label_encoder,
            "task": task,
            "model_type": model_type,
            "feature_columns": feature_cols,
            "target_column": target_col,
            "metrics": metrics,
            "random_seed": seed,
            "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        joblib.dump(artifact_bundle, artifact_file)

        # Write metadata.json
        meta_file = os.path.join(user_models_dir, "metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "job_id": job_id,
                "user_id": user_id,
                "dataset_id": dataset.id,
                "task": task,
                "model_type": model_type,
                "feature_columns": feature_cols,
                "target_column": target_col,
                "metrics": metrics,
                "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, f, indent=2)

        # 6. Complete
        job.status = "completed"
        job.current_step = "Training completed successfully"
        job.progress = 100.0
        job.artifact_path = artifact_file
        job.metrics_json = json.dumps(metrics)
        job.completed_at = datetime.datetime.now(datetime.timezone.utc)

        current_logs = json.loads(job.logs_json) if job.logs_json else []
        current_logs.append(f"[{datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}] Model artifact saved. Status: COMPLETED.")
        job.logs_json = json.dumps(current_logs)
        db.commit()

    except Exception as exc:
        logger.exception("Error executing training job %s: %s", job_id, exc)
        try:
            job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
            if job and job.status != "cancelled":
                job.status = "failed"
                job.current_step = "Training failed"
                job.error_message = f"Training error: {str(exc)}"
                job.completed_at = datetime.datetime.now(datetime.timezone.utc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def serialize_training_job(job: TrainingJobRecord, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Formats TrainingJobRecord to match the frontend TrainingJob interface."""
    features = json.loads(job.feature_columns_json) if job.feature_columns_json else []
    metrics = json.loads(job.metrics_json) if job.metrics_json else None
    logs = json.loads(job.logs_json) if job.logs_json else []

    config_dict = {
        "dataset_id": job.dataset_id,
        "task": job.task,
        "target_column": job.target_column,
        "feature_columns": features,
        "model_type": job.model_type,
        "train_split": job.train_split,
        "validation_split": job.validation_split,
        "test_split": job.test_split,
        "random_seed": job.random_seed,
    }

    duration_seconds = None
    if job.started_at and job.completed_at:
        try:
            duration_seconds = round((job.completed_at - job.started_at).total_seconds(), 2)
        except Exception:
            pass

    return {
        "id": job.id,
        "dataset_id": job.dataset_id,
        "dataset_name": dataset_name or (job.dataset.name if job.dataset else None),
        "task": job.task,
        "model_type": job.model_type,
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "logs": logs,
        "error_message": job.error_message,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "duration_seconds": duration_seconds,
        "created_at": job.created_at.isoformat() if job.created_at else "",
        "config": config_dict,
        "metrics": metrics,
        "artifactPath": job.artifact_path,
    }
