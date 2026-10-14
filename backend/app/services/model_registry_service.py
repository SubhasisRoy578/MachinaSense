"""Model Registry Service for MachinaSense Backend Part 3.

Provides registration, versioning, atomic activation, and retrieval for trained models.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ModelRegistryRecord, TrainingJobRecord, User

logger = logging.getLogger("machinasense.registry")

SUPPORTED_TASKS = {"rul_regression", "condition_classification", "anomaly_detection"}


def sanitize_path_safety(artifact_path: Optional[str]) -> str:
    """Verifies artifact path exists, has no path traversal, and resides within MODELS_DIR."""
    if not artifact_path or not isinstance(artifact_path, str) or not artifact_path.strip():
        raise HTTPException(status_code=400, detail="Model artifact path is missing or invalid.")

    # Guard against traversal characters
    if ".." in artifact_path:
        raise HTTPException(status_code=400, detail="Path traversal in artifact path is not permitted.")

    norm_path = os.path.abspath(artifact_path)
    base_dir = os.path.abspath(settings.MODELS_DIR)

    # Path must reside within the authorized models base directory
    try:
        common = os.path.commonpath([norm_path, base_dir])
    except ValueError:
        raise HTTPException(status_code=400, detail="Artifact path is outside authorized storage.")

    if common != base_dir:
        raise HTTPException(status_code=400, detail="Artifact path is outside authorized storage.")

    if not os.path.exists(norm_path):
        raise HTTPException(status_code=400, detail="Model artifact file does not exist on disk.")

    if not os.path.isfile(norm_path) or not os.access(norm_path, os.R_OK):
        raise HTTPException(status_code=400, detail="Model artifact file is not readable.")

    return norm_path


def serialize_registered_model(record: ModelRegistryRecord, include_metadata: bool = True) -> Dict[str, Any]:
    """Formats ModelRegistryRecord for API responses without exposing internal disk paths."""
    metrics = json.loads(record.metrics_json) if record.metrics_json else {}
    result: Dict[str, Any] = {
        "id": record.id,
        "name": record.name,
        "task": record.task,
        "model_type": record.model_type,
        "version": record.version,
        "status": record.status,
        "training_job_id": record.training_job_id,
        "dataset_id": record.dataset_id,
        "metrics": metrics,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "activated_at": record.activated_at.isoformat() if record.activated_at else None,
        "deactivated_at": record.deactivated_at.isoformat() if record.deactivated_at else None,
    }
    if include_metadata:
        metadata = json.loads(record.training_metadata_json) if record.training_metadata_json else {}
        result["training_metadata"] = metadata
    return result


def register_model(db: Session, user: User, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Registers a completed training job as a versioned model in the registry."""
    job_id = payload.get("training_job_id")
    raw_name = payload.get("name")

    if not job_id or not isinstance(job_id, str) or not job_id.strip():
        raise HTTPException(status_code=400, detail="training_job_id is required.")

    if not raw_name or not isinstance(raw_name, str) or not raw_name.strip():
        raise HTTPException(status_code=400, detail="Model name is required and cannot be empty.")

    name = raw_name.strip()
    if len(name) > 255:
        raise HTTPException(status_code=400, detail="Model name cannot exceed 255 characters.")

    # 1. Verify training job ownership
    job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id.strip()).first()
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Training job not found.")

    # 2. Verify status == completed
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot register model: training job status is '{job.status}' (must be 'completed').",
        )

    # 3. Verify task is supported
    if job.task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail=f"Unsupported task '{job.task}'.")

    # 4. Verify artifact path and safety
    sanitize_path_safety(job.artifact_path)

    # 5. Verify evaluation metrics exist and are valid
    if not job.metrics_json or not job.metrics_json.strip():
        raise HTTPException(status_code=400, detail="Evaluation metrics are missing from training job.")

    try:
        metrics = json.loads(job.metrics_json)
        if not metrics or not isinstance(metrics, dict):
            raise HTTPException(status_code=400, detail="Evaluation metrics are empty or invalid.")
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Evaluation metrics JSON is corrupted.") from exc

    # 6. Verify training job has not already been registered
    existing_reg = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.training_job_id == job.id).first()
    if existing_reg:
        raise HTTPException(
            status_code=409,
            detail=f"Training job '{job.id}' is already registered as model '{existing_reg.name}' (version {existing_reg.version}).",
        )

    # 7. Determine next version for this user + model name
    existing_versions = (
        db.query(ModelRegistryRecord)
        .filter(ModelRegistryRecord.user_id == user.id, ModelRegistryRecord.name == name)
        .all()
    )
    if not existing_versions:
        next_version = 1
    else:
        next_version = max(m.version for m in existing_versions) + 1

    # 8. Build training metadata
    features = json.loads(job.feature_columns_json) if job.feature_columns_json else []
    training_metadata = {
        "dataset_id": job.dataset_id,
        "task": job.task,
        "model_type": job.model_type,
        "target_column": job.target_column,
        "feature_columns": features,
        "train_split": job.train_split,
        "validation_split": job.validation_split,
        "test_split": job.test_split,
        "random_seed": job.random_seed,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }

    # 9. Create record
    model_id = f"model_{uuid.uuid4().hex[:12]}"
    record = ModelRegistryRecord(
        id=model_id,
        user_id=user.id,
        training_job_id=job.id,
        dataset_id=job.dataset_id,
        name=name,
        task=job.task,
        model_type=job.model_type,
        version=next_version,
        status="inactive",
        artifact_path=job.artifact_path,
        metrics_json=json.dumps(metrics),
        training_metadata_json=json.dumps(training_metadata),
        created_at=datetime.now(timezone.utc),
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info("Registered model %s (name='%s', v%d) for user %s", model_id, name, next_version, user.id)
    return serialize_registered_model(record, include_metadata=True)


def list_models(
    db: Session,
    user: User,
    task: Optional[str] = None,
    status: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Lists registered models belonging to the authenticated user with optional filters."""
    query = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.user_id == user.id)

    if task:
        query = query.filter(ModelRegistryRecord.task == task)
    if status:
        query = query.filter(ModelRegistryRecord.status == status)
    if name:
        query = query.filter(ModelRegistryRecord.name == name)

    records = query.order_by(ModelRegistryRecord.created_at.desc()).all()
    return {
        "models": [serialize_registered_model(r, include_metadata=False) for r in records],
        "total": len(records),
    }


def get_model_detail(db: Session, user: User, model_id: str) -> Dict[str, Any]:
    """Retrieves full model details with metadata, scoped to authenticated user."""
    record = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == model_id).first()
    if not record or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Model not found.")

    return serialize_registered_model(record, include_metadata=True)


def get_model_versions(db: Session, user: User, model_id: str) -> List[Dict[str, Any]]:
    """Retrieves all versions of the logical model belonging to the authenticated user."""
    target = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == model_id).first()
    if not target or target.user_id != user.id:
        raise HTTPException(status_code=404, detail="Model not found.")

    versions = (
        db.query(ModelRegistryRecord)
        .filter(
            ModelRegistryRecord.user_id == user.id,
            ModelRegistryRecord.name == target.name,
            ModelRegistryRecord.task == target.task,
        )
        .order_by(ModelRegistryRecord.version.desc())
        .all()
    )

    return [serialize_registered_model(v, include_metadata=False) for v in versions]


def activate_model(db: Session, user: User, model_id: str) -> Dict[str, Any]:
    """Atomically activates a model and deactivates any existing active model for the user+task."""
    model = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == model_id).first()
    if not model or model.user_id != user.id:
        raise HTTPException(status_code=404, detail="Model not found.")

    if model.task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=400, detail=f"Unsupported task '{model.task}'.")

    # Verify artifact is intact and readable before activation
    sanitize_path_safety(model.artifact_path)

    # Verify origin job
    if not model.training_job or model.training_job.status != "completed":
        raise HTTPException(status_code=400, detail="Model origin training job is not completed.")

    now_ts = datetime.now(timezone.utc)

    # Atomic swap: deactivate currently active models for this user + task
    currently_active = (
        db.query(ModelRegistryRecord)
        .filter(
            ModelRegistryRecord.user_id == user.id,
            ModelRegistryRecord.task == model.task,
            ModelRegistryRecord.status == "active",
            ModelRegistryRecord.id != model.id,
        )
        .all()
    )

    for active_item in currently_active:
        active_item.status = "inactive"
        active_item.deactivated_at = now_ts

    model.status = "active"
    model.activated_at = now_ts
    model.deactivated_at = None

    db.commit()
    db.refresh(model)

    logger.info("Activated model %s (task=%s, v%d) for user %s", model.id, model.task, model.version, user.id)
    return serialize_registered_model(model, include_metadata=True)


def deactivate_model(db: Session, user: User, model_id: str) -> Dict[str, Any]:
    """Deactivates a model if active; idempotent if already inactive."""
    model = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == model_id).first()
    if not model or model.user_id != user.id:
        raise HTTPException(status_code=404, detail="Model not found.")

    if model.status == "active":
        model.status = "inactive"
        model.deactivated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(model)
        logger.info("Deactivated model %s for user %s", model.id, user.id)

    return serialize_registered_model(model, include_metadata=True)


def get_active_model_for_task(db: Session, user: User, task: str) -> Dict[str, Any]:
    """Returns the currently active user model for the specified task."""
    if task not in SUPPORTED_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Supported tasks: {', '.join(sorted(SUPPORTED_TASKS))}.",
        )

    active_model = (
        db.query(ModelRegistryRecord)
        .filter(
            ModelRegistryRecord.user_id == user.id,
            ModelRegistryRecord.task == task,
            ModelRegistryRecord.status == "active",
        )
        .first()
    )

    if not active_model:
        raise HTTPException(status_code=404, detail=f"No active model for task '{task}'.")

    return serialize_registered_model(active_model, include_metadata=True)
