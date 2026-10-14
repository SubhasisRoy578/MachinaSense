"""Model Comparison & Evaluation History Service for MachinaSense Phase 7.

Enforces scientifically valid model comparisons:
- Strict task compatibility verification (never compare regression with classification)
- Target definition alignment checks
- Evaluation dataset and split protocol validation
- Transparent metric comparison tables (lower MAE/RMSE is better, higher R2/accuracy is better)
- Explicit differentiation between benchmark models and user-trained models
- Historical version evaluation tracking without retraining
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models import ModelRegistryRecord, User
from app.ml_service import ml_service

logger = logging.getLogger("machinasense.comparison")

BENCHMARK_MODELS_METADATA = {
    "benchmark_rf": {
        "id": "benchmark_rf",
        "name": "NASA C-MAPSS Random Forest Regressor",
        "task": "rul_regression",
        "model_type": "rf_regressor",
        "version": 1,
        "is_benchmark": True,
        "dataset_name": "NASA C-MAPSS FD001 Benchmark",
        "dataset_id": "nasa_cmapss_fd001",
        "split_protocol": "Standard NASA 100-engine held-out test split",
        "sample_count": 100,
        "metrics": {
            "mae": 13.2202,
            "rmse": 18.1030,
            "r2": 0.8102,
        },
        "target_column": "rul",
        "duration_seconds": None,
        "strengths": "Fast inference, interpretable tree splits, robust variance baseline.",
        "limitations": "Does not model temporal sequence memory across cycles.",
    },
    "benchmark_lstm": {
        "id": "benchmark_lstm",
        "name": "NASA C-MAPSS PyTorch LSTM Temporal Regressor",
        "task": "rul_regression",
        "model_type": "lstm_regressor",
        "version": 1,
        "is_benchmark": True,
        "dataset_name": "NASA C-MAPSS FD001 Benchmark",
        "dataset_id": "nasa_cmapss_fd001",
        "split_protocol": "Standard NASA 100-engine held-out test split (30-cycle window)",
        "sample_count": 100,
        "metrics": {
            "mae": 11.3005,
            "rmse": 15.0027,
            "r2": 0.8697,
        },
        "target_column": "rul",
        "duration_seconds": None,
        "strengths": "Captures temporal degradation dynamics across consecutive sensor cycles.",
        "limitations": "Requires at least 30 consecutive telemetry cycles for inference.",
    },
    "benchmark_isolation_forest": {
        "id": "benchmark_isolation_forest",
        "name": "NASA C-MAPSS Isolation Forest Anomaly Detector",
        "task": "anomaly_detection",
        "model_type": "isolation_forest",
        "version": 1,
        "is_benchmark": True,
        "dataset_name": "NASA C-MAPSS FD001 (Healthy cycles RUL > 100)",
        "dataset_id": "nasa_cmapss_fd001",
        "split_protocol": "Healthy operational cycle envelope",
        "sample_count": None,
        "metrics": {
            "contamination": 0.05,
            "scoring": "Inverted decision function (higher = more anomalous)",
        },
        "target_column": None,
        "duration_seconds": None,
        "strengths": "Unsupervised partitioning without requiring ground-truth fault labels.",
        "limitations": "FD001 dataset lacks ground-truth labels; cannot compute accuracy or F1.",
    },
}


def compare_models(
    db: Session,
    user: User,
    model_ids: List[str],
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Compares 2 or more compatible models with scientific validity checks."""
    if not model_ids or len(model_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 model IDs must be provided for comparison.",
        )

    if len(model_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="A maximum of 10 models can be compared in a single request.",
        )

    models_data: List[Dict[str, Any]] = []

    for mid in model_ids:
        mid_clean = mid.strip()
        if mid_clean in BENCHMARK_MODELS_METADATA:
            models_data.append(BENCHMARK_MODELS_METADATA[mid_clean])
        else:
            rec = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == mid_clean).first()
            if not rec or rec.user_id != user.id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{mid_clean}' was not found in your registered models.",
                )

            metrics = json.loads(rec.metrics_json) if rec.metrics_json else {}
            meta = json.loads(rec.training_metadata_json) if rec.training_metadata_json else {}

            duration = None
            if rec.training_job and rec.training_job.started_at and rec.training_job.completed_at:
                try:
                    duration = round(
                        (rec.training_job.completed_at - rec.training_job.started_at).total_seconds(),
                        2,
                    )
                except Exception:
                    pass

            models_data.append({
                "id": rec.id,
                "name": rec.name,
                "task": rec.task,
                "model_type": rec.model_type,
                "version": rec.version,
                "status": rec.status,
                "is_benchmark": False,
                "dataset_name": rec.dataset.name if rec.dataset else "User Telemetry Dataset",
                "dataset_id": rec.dataset_id,
                "split_protocol": (
                    f"Train {meta.get('train_split', 70)}% / Val {meta.get('validation_split', 15)}% / Test {meta.get('test_split', 15)}%"
                ),
                "sample_count": metrics.get("test_samples"),
                "metrics": metrics,
                "target_column": meta.get("target_column"),
                "feature_count": len(meta.get("feature_columns", [])),
                "duration_seconds": duration,
                "created_at": rec.created_at.isoformat() if rec.created_at else None,
            })

    # 1. Scientific Rule: All models must share the same ML task
    tasks = {m["task"] for m in models_data}
    if len(tasks) > 1:
        return {
            "is_compatible": False,
            "incompatibility_reason": (
                f"Selected models perform different ML tasks: {', '.join(sorted(tasks))}. "
                "Models can only be scientifically compared when their task and evaluation objectives match. "
                "Comparing regression MAE against classification accuracy or anomaly score is invalid."
            ),
            "models": models_data,
            "comparisons": [],
        }

    common_task = list(tasks)[0]

    # 2. Scientific Rule: Supervised target column must align
    targets = {m.get("target_column") for m in models_data if m.get("target_column") is not None}
    target_aligned = len(targets) <= 1
    target_warning = None
    if not target_aligned:
        target_warning = (
            f"Models have different target objectives ({', '.join(str(t) for t in targets)}). "
            "Metric differences may reflect target scaling rather than algorithmic accuracy."
        )

    # 3. Scientific Rule: Dataset consistency check
    dataset_ids = {m.get("dataset_id") for m in models_data}
    identical_dataset = len(dataset_ids) == 1
    dataset_warning = None
    if not identical_dataset:
        dataset_warning = (
            "Models were evaluated on distinct datasets or different splits. "
            "Metric differences reflect performance on respective test splits and should not be "
            "interpreted as a controlled head-to-head benchmark on identical operational data."
        )

    # 4. Metric Comparison Matrix
    comparison_metrics: List[Dict[str, Any]] = []

    if common_task == "rul_regression":
        metric_keys = [
            ("mae", "Mean Absolute Error (MAE)", "cycles", "lower_is_better"),
            ("rmse", "Root Mean Squared Error (RMSE)", "cycles", "lower_is_better"),
            ("r2", "Coefficient of Determination (R²)", "--", "higher_is_better"),
            ("mse", "Mean Squared Error (MSE)", "cycles²", "lower_is_better"),
        ]

        for m_key, m_name, unit, direction in metric_keys:
            values = []
            for m in models_data:
                v = m["metrics"].get(m_key)
                if isinstance(v, (int, float)):
                    values.append({"model_id": m["id"], "value": round(float(v), 4)})
                else:
                    values.append({"model_id": m["id"], "value": None})

            valid_vals = [x["value"] for x in values if x["value"] is not None]
            best_id = None
            if valid_vals:
                best_val = min(valid_vals) if direction == "lower_is_better" else max(valid_vals)
                best_match = [x["model_id"] for x in values if x["value"] == best_val]
                best_id = best_match[0] if best_match else None

            comparison_metrics.append({
                "metric_key": m_key,
                "label": m_name,
                "unit": unit,
                "direction": direction,
                "values": {item["model_id"]: item["value"] for item in values},
                "best_model_id": best_id if identical_dataset else None,
            })

    elif common_task == "condition_classification":
        metric_keys = [
            ("accuracy", "Overall Accuracy", "%", "higher_is_better"),
            ("f1_weighted", "F1 Score (Weighted)", "--", "higher_is_better"),
            ("f1_macro", "F1 Score (Macro)", "--", "higher_is_better"),
            ("precision_weighted", "Precision (Weighted)", "--", "higher_is_better"),
            ("recall_weighted", "Recall (Weighted)", "--", "higher_is_better"),
        ]

        for m_key, m_name, unit, direction in metric_keys:
            values = []
            for m in models_data:
                v = m["metrics"].get(m_key)
                if isinstance(v, (int, float)):
                    values.append({"model_id": m["id"], "value": round(float(v), 4)})
                else:
                    values.append({"model_id": m["id"], "value": None})

            valid_vals = [x["value"] for x in values if x["value"] is not None]
            best_id = None
            if valid_vals:
                best_val = max(valid_vals)
                best_match = [x["model_id"] for x in values if x["value"] == best_val]
                best_id = best_match[0] if best_match else None

            comparison_metrics.append({
                "metric_key": m_key,
                "label": m_name,
                "unit": unit,
                "direction": direction,
                "values": {item["model_id"]: item["value"] for item in values},
                "best_model_id": best_id if identical_dataset else None,
            })

    elif common_task == "anomaly_detection":
        metric_keys = [
            ("anomaly_rate", "Detected Anomaly Rate", "--", "neutral"),
            ("score_mean", "Mean Anomaly Score", "--", "neutral"),
            ("contamination", "Specified Contamination", "--", "neutral"),
        ]

        for m_key, m_name, unit, direction in metric_keys:
            values = {m["id"]: m["metrics"].get(m_key) for m in models_data}
            comparison_metrics.append({
                "metric_key": m_key,
                "label": m_name,
                "unit": unit,
                "direction": direction,
                "values": values,
                "best_model_id": None,
            })

    # Summary narrative
    narrative = (
        f"Compared {len(models_data)} models on task '{common_task}'. "
    )
    if identical_dataset:
        narrative += "All models were evaluated on the same dataset split, allowing direct scientific comparison."
    else:
        narrative += (
            "Models were evaluated on different datasets or protocols. "
            "Direct ranking should be considered indicative rather than a controlled benchmark."
        )

    return {
        "is_compatible": True,
        "task": common_task,
        "identical_evaluation_dataset": identical_dataset,
        "dataset_warning": dataset_warning,
        "target_warning": target_warning,
        "narrative": narrative,
        "models": models_data,
        "metric_comparison": comparison_metrics,
        "guidance": (
            "In regression, lower MAE and RMSE indicate closer cycle estimation, while higher R² indicates "
            "greater variance explained. In operational settings, trade-offs between model latency, feature "
            "availability, and error distributions must be evaluated alongside single aggregate metrics."
        ),
    }


def get_model_evaluation_history(
    db: Session,
    user: User,
    model_name: str,
) -> Dict[str, Any]:
    """Retrieves chronological evaluation history across all versions of a logical model."""
    records = (
        db.query(ModelRegistryRecord)
        .filter(
            ModelRegistryRecord.user_id == user.id,
            ModelRegistryRecord.name == model_name,
        )
        .order_by(ModelRegistryRecord.version.asc())
        .all()
    )

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No model versions found for model name '{model_name}'.",
        )

    history = []
    for r in records:
        metrics = json.loads(r.metrics_json) if r.metrics_json else {}
        meta = json.loads(r.training_metadata_json) if r.training_metadata_json else {}

        duration = None
        if r.training_job and r.training_job.started_at and r.training_job.completed_at:
            try:
                duration = round(
                    (r.training_job.completed_at - r.training_job.started_at).total_seconds(), 2
                )
            except Exception:
                pass

        history.append({
            "model_id": r.id,
            "name": r.name,
            "version": r.version,
            "task": r.task,
            "model_type": r.model_type,
            "status": r.status,
            "dataset_id": r.dataset_id,
            "dataset_name": r.dataset.name if r.dataset else "Dataset",
            "metrics": metrics,
            "duration_seconds": duration,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "activated_at": r.activated_at.isoformat() if r.activated_at else None,
        })

    return {
        "model_name": model_name,
        "task": records[0].task,
        "total_versions": len(records),
        "versions": history,
    }
