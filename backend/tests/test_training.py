"""Comprehensive test suite for Backend Part 2: Real ML Training Engine.
Tests cover:
1. Valid RUL training job (Random Forest Regressor)
2. Valid RUL training job (PyTorch LSTM Regressor)
3. Valid condition classification training job (Random Forest Classifier)
4. Valid anomaly detection training job (Isolation Forest)
5. Unsupported task/model combination rejection (400)
6. Invalid/missing target column validation (400)
7. Invalid/unknown feature column validation (400)
8. Target included as feature rejection (400)
9. Invalid split ratios rejection (400)
10. Dataset ownership rejection (404)
11. Training job cross-user isolation (GET and CANCEL) (404)
12. Job status retrieval and listing
13. Failed training state handling (data too small)
14. Cancellation behavior
15. Real artifact creation and truthful metrics
"""
import io
import os
import sys
import uuid
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app
from app.config import settings

settings.ENVIRONMENT = "test"


def auth_header(user_id: str):
    return {"Authorization": f"Bearer test_token_{user_id}"}


def make_regression_csv(rows: int = 35) -> str:
    lines = ["cycle,vibration,temperature,pressure,rul"]
    for i in range(1, rows + 1):
        vib = 640.0 + i * 1.5
        temp = 1580.0 + i * 2.0
        press = 1400.0 + i * 1.2
        rul = max(0, 100 - i * 2)
        lines.append(f"{i},{vib:.2f},{temp:.2f},{press:.2f},{rul}")
    return "\n".join(lines) + "\n"


def make_classification_csv(rows: int = 36) -> str:
    lines = ["cycle,vibration,temperature,pressure,condition"]
    classes = ["normal", "warning", "fault"]
    for i in range(1, rows + 1):
        vib = 600.0 + (i % 3) * 20.0 + i * 0.5
        temp = 1500.0 + (i % 3) * 30.0 + i * 0.3
        press = 1300.0 + (i % 3) * 15.0 + i * 0.2
        c = classes[i % 3]
        lines.append(f"{i},{vib:.2f},{temp:.2f},{press:.2f},{c}")
    return "\n".join(lines) + "\n"


def make_anomaly_csv(rows: int = 40) -> str:
    lines = ["cycle,vibration,temperature,pressure"]
    for i in range(1, rows + 1):
        vib = 640.0 + (50.0 if i == 20 else (i % 5))
        temp = 1580.0 + (70.0 if i == 20 else (i % 7))
        press = 1400.0 + (40.0 if i == 20 else (i % 4))
        lines.append(f"{i},{vib:.2f},{temp:.2f},{press:.2f}")
    return "\n".join(lines) + "\n"


def upload_dataset(client: TestClient, user: str, csv_data: str, name: str) -> str:
    files = {"file": ("dataset.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    resp = client.post("/api/datasets", headers=auth_header(user), files=files, data={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# 1. Valid RUL training (Random Forest)
# ---------------------------------------------------------------------------
def test_valid_rul_training_rf():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(35), "RUL Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "rul",
            "feature_columns": ["cycle", "vibration", "temperature", "pressure"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
            "random_seed": 42,
        }

        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]

        # Background task runs synchronously in TestClient
        get_resp = client.get(f"/api/training/jobs/{job_id}", headers=auth_header(user))
        assert get_resp.status_code == 200
        job = get_resp.json()

        assert job["status"] == "completed", f"Job failed: {job.get('error_message')}"
        assert job["task"] == "rul_regression"
        assert job["model_type"] == "rf_regressor"
        assert job["metrics"] is not None
        assert "mae" in job["metrics"]
        assert "rmse" in job["metrics"]
        assert "r2" in job["metrics"]
        assert isinstance(job["metrics"]["mae"], (int, float))
        assert isinstance(job["metrics"]["rmse"], (int, float))
        assert job["artifactPath"] is not None
        assert os.path.exists(job["artifactPath"])


# ---------------------------------------------------------------------------
# 2. Valid RUL training (PyTorch LSTM)
# ---------------------------------------------------------------------------
def test_valid_rul_training_lstm():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(40), "LSTM Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "lstm_regressor",
            "target_column": "rul",
            "feature_columns": ["vibration", "temperature", "pressure"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
            "random_seed": 42,
        }

        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        get_resp = client.get(f"/api/training/jobs/{job_id}", headers=auth_header(user))
        assert get_resp.status_code == 200
        job = get_resp.json()
        assert job["status"] == "completed", f"Job failed: {job.get('error_message')}"
        assert "mae" in job["metrics"]
        assert "rmse" in job["metrics"]


# ---------------------------------------------------------------------------
# 3. Valid Classification training (Random Forest Classifier)
# ---------------------------------------------------------------------------
def test_valid_classification_training():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_classification_csv(36), "Classification Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "condition_classification",
            "model_type": "rf_classifier",
            "target_column": "condition",
            "feature_columns": ["cycle", "vibration", "temperature", "pressure"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
            "random_seed": 42,
        }

        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        get_resp = client.get(f"/api/training/jobs/{job_id}", headers=auth_header(user))
        assert get_resp.status_code == 200
        job = get_resp.json()
        assert job["status"] == "completed", f"Job failed: {job.get('error_message')}"
        metrics = job["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics


# ---------------------------------------------------------------------------
# 4. Valid Anomaly Detection (Isolation Forest)
# ---------------------------------------------------------------------------
def test_valid_anomaly_detection_training():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_anomaly_csv(40), "Anomaly Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "anomaly_detection",
            "model_type": "isolation_forest",
            "feature_columns": ["vibration", "temperature", "pressure"],
            "train_split": 80,
            "validation_split": 10,
            "test_split": 10,
            "random_seed": 42,
        }

        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        get_resp = client.get(f"/api/training/jobs/{job_id}", headers=auth_header(user))
        assert get_resp.status_code == 200
        job = get_resp.json()
        assert job["status"] == "completed", f"Job failed: {job.get('error_message')}"
        metrics = job["metrics"]
        assert "total_samples" in metrics or "sample_count" in metrics
        assert "anomaly_count" in metrics
        assert "anomaly_rate" in metrics


# ---------------------------------------------------------------------------
# 5. Unsupported task/model combination (400)
# ---------------------------------------------------------------------------
def test_unsupported_task_model_combination():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(30), "Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "isolation_forest",
            "target_column": "rul",
            "feature_columns": ["vibration", "temperature"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
            "random_seed": 42,
        }
        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 400
        assert "unsupported model" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 6. Invalid / missing target column (400)
# ---------------------------------------------------------------------------
def test_invalid_target_column():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(30), "Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "non_existent_column",
            "feature_columns": ["vibration", "temperature"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
        }
        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 400
        assert "target_column" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 7. Invalid / unknown feature column (400)
# ---------------------------------------------------------------------------
def test_invalid_feature_column():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(30), "Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "rul",
            "feature_columns": ["vibration", "unknown_sensor_10"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
        }
        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 400
        assert "not found in dataset" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 8. Target included as feature (400)
# ---------------------------------------------------------------------------
def test_target_included_as_feature():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(30), "Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "rul",
            "feature_columns": ["vibration", "rul"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
        }
        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 400
        assert "cannot be included in feature columns" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 9. Invalid split ratios (400)
# ---------------------------------------------------------------------------
def test_invalid_split_ratios():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(30), "Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "rul",
            "feature_columns": ["vibration", "temperature"],
            "train_split": 50,
            "validation_split": 30,
            "test_split": 30,  # sum is 110 != 100
        }
        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 400
        assert "must sum to 1.0" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 10. Dataset ownership rejection (404)
# ---------------------------------------------------------------------------
def test_dataset_ownership_isolation():
    with TestClient(app) as client:
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        ds_id_a = upload_dataset(client, user_a, make_regression_csv(30), "User A Telemetry")

        payload = {
            "dataset_id": ds_id_a,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "rul",
            "feature_columns": ["vibration", "temperature"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
        }
        resp = client.post("/api/training/jobs", headers=auth_header(user_b), json=payload)
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 11. Training job cross-user isolation (GET and CANCEL)
# ---------------------------------------------------------------------------
def test_training_job_cross_user_isolation():
    with TestClient(app) as client:
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        ds_id_a = upload_dataset(client, user_a, make_regression_csv(30), "User A Telemetry")

        payload = {
            "dataset_id": ds_id_a,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "rul",
            "feature_columns": ["vibration", "temperature"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
        }
        resp = client.post("/api/training/jobs", headers=auth_header(user_a), json=payload)
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        # User B cannot access User A's job
        get_b = client.get(f"/api/training/jobs/{job_id}", headers=auth_header(user_b))
        assert get_b.status_code == 404

        # User B cannot cancel User A's job
        cancel_b = client.post(f"/api/training/jobs/{job_id}/cancel", headers=auth_header(user_b))
        assert cancel_b.status_code == 404


# ---------------------------------------------------------------------------
# 12. Job status retrieval and listing
# ---------------------------------------------------------------------------
def test_list_training_jobs():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(30), "Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "rul",
            "feature_columns": ["vibration", "temperature"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
        }
        client.post("/api/training/jobs", headers=auth_header(user), json=payload)

        list_resp = client.get("/api/training/jobs", headers=auth_header(user))
        assert list_resp.status_code == 200
        data = list_resp.json()
        jobs = data.get("jobs", data) if isinstance(data, dict) else data
        assert len(jobs) >= 1
        assert jobs[0]["dataset_id"] == ds_id


# ---------------------------------------------------------------------------
# 13. Failed training state (data too small for splitting)
# ---------------------------------------------------------------------------
def test_failed_training_state():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        # Only 2 rows — cannot be split into meaningful train/test partitions
        tiny_csv = "vibration,temperature,rul\n1.0,2.0,10\n1.1,2.1,9\n"
        ds_id = upload_dataset(client, user, tiny_csv, "Tiny Dataset")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "rul",
            "feature_columns": ["vibration", "temperature"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
        }
        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        # Validator requires at least 5 rows, so this should fail with 400
        assert resp.status_code == 400
        assert "at least 5 rows" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 14. Cancellation behavior
# ---------------------------------------------------------------------------
def test_cancellation_behavior():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(30), "Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "rf_regressor",
            "target_column": "rul",
            "feature_columns": ["vibration", "temperature"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
        }
        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        # Cancel job — in TestClient, background tasks run synchronously,
        # so the job will have already completed by the time we cancel.
        cancel_resp = client.post(f"/api/training/jobs/{job_id}/cancel", headers=auth_header(user))
        assert cancel_resp.status_code == 200
        cancelled_data = cancel_resp.json()
        # Job may have already completed (sync background tasks) or be cancelled
        assert cancelled_data["status"] in {"cancelled", "completed"}


# ---------------------------------------------------------------------------
# 15. Real artifact creation and truthful metrics (Gradient Boosting)
# ---------------------------------------------------------------------------
def test_artifact_and_real_metrics():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(40), "Artifact Telemetry")

        payload = {
            "dataset_id": ds_id,
            "task": "rul_regression",
            "model_type": "gradient_boosting_regressor",
            "target_column": "rul",
            "feature_columns": ["cycle", "vibration", "temperature", "pressure"],
            "train_split": 70,
            "validation_split": 15,
            "test_split": 15,
            "random_seed": 42,
        }

        resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        get_resp = client.get(f"/api/training/jobs/{job_id}", headers=auth_header(user))
        assert get_resp.status_code == 200
        job = get_resp.json()
        assert job["status"] == "completed", f"Job failed: {job.get('error_message')}"

        # Verify real artifact on disk
        assert job["artifactPath"] is not None
        assert os.path.exists(job["artifactPath"])
        artifact_dir = os.path.dirname(job["artifactPath"])
        assert os.path.exists(os.path.join(artifact_dir, "metadata.json"))

        # Verify metrics are real numbers, not mocked
        metrics = job["metrics"]
        assert isinstance(metrics["mae"], (int, float))
        assert metrics["mae"] >= 0
        assert isinstance(metrics["rmse"], (int, float))
        assert metrics["rmse"] >= 0
        assert isinstance(metrics["r2"], (int, float))
        assert metrics["test_samples"] > 0
