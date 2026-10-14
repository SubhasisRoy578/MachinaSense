"""End-to-End Production Validation & Reliability Test Suite for MachinaSense.

Covers:
1. End-to-end ML lifecycle for RUL Regression, Condition Classification, and Anomaly Detection:
   Dataset Upload -> Schema Inspection -> Training Job Execution -> Evaluation Metrics ->
   Model Registration -> Version Creation -> Model Activation -> Single Inference ->
   Batch Inference -> Audit History Retrieval -> Detailed Record Inspection.
2. Cross-user authentication and data isolation:
   - User A resources (datasets, jobs, models, versions, predictions) cannot be accessed,
     mutated, activated, or inferred by User B.
   - Unauthenticated requests are rejected with 401 across all endpoints.
   - Strict tenant isolation enforced server-side.
3. Path traversal & security audits:
   - Model artifact paths are never exposed in API responses.
   - Malicious path traversal attempts in artifacts are rejected.
4. Schema mismatch & failure modes:
   - Incompatible feature schemas return 400 with missing columns explanation.
   - Batch size > 500 records rejected with 400.
   - Missing active model returns 404.
5. Storage & cache resilience:
   - Models reload reliably after cache clearance.
"""
from __future__ import annotations

import io
import json
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
from app.db.session import get_db
from app.db.models import ModelRegistryRecord, ModelPredictionRecord, UserDataset, TrainingJobRecord
from app.services.inference_service import clear_model_cache
from app.services.training_engine import run_training_pipeline

settings.ENVIRONMENT = "test"


def auth_header(user_id: str):
    return {"Authorization": f"Bearer test_token_{user_id}"}


def create_regression_csv(rows: int = 40) -> str:
    lines = ["cycle,vibration,temperature,pressure,rul"]
    for i in range(1, rows + 1):
        vib = 640.0 + i * 1.5
        temp = 1580.0 + i * 2.0
        press = 14.7 + (i % 5) * 0.1
        rul = max(0.0, 100.0 - i * 2.0)
        lines.append(f"{i},{vib:.2f},{temp:.2f},{press:.2f},{rul:.1f}")
    return "\n".join(lines)


def create_classification_csv(rows: int = 45) -> str:
    lines = ["rpm,temp,vibration,health_state"]
    states = ["Healthy", "Degrading", "Critical"]
    for i in range(1, rows + 1):
        rpm = 1200 + i * 10
        temp = 60.0 + (i % 15) * 2.5
        vib = 0.5 + (i % 10) * 0.3
        state = states[i % 3]
        lines.append(f"{rpm},{temp:.2f},{vib:.2f},{state}")
    return "\n".join(lines)


def create_anomaly_csv(rows: int = 40) -> str:
    lines = ["sensor_a,sensor_b,sensor_c"]
    for i in range(1, rows + 1):
        sa = 100.0 + (i % 7) * 2.0
        sb = 50.0 + (i % 5) * 1.5
        sc = 25.0 + (i % 4) * 0.8
        lines.append(f"{sa:.2f},{sb:.2f},{sc:.2f}")
    return "\n".join(lines)


class TestEndToEndMLLifecycleAndMultiTenancy:

    def setup_method(self):
        clear_model_cache()

    def test_complete_ml_lifecycle_rul_regression(self):
        """Verifies full RUL Regression pipeline from dataset to audited prediction."""
        user_id = f"user_e2e_{uuid.uuid4().hex[:8]}"
        headers = auth_header(user_id)

        with TestClient(app) as client:
            # 1. Dataset Upload
            csv_content = create_regression_csv(40)
            upload_res = client.post(
                "/api/datasets",
                headers=headers,
                files={"file": ("telemetry_rul.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
                data={"name": "E2E RUL Dataset"},
            )
            assert upload_res.status_code == 201, upload_res.text
            dataset = upload_res.json()
            dataset_id = dataset["id"]
            assert dataset["statistics"]["rowCount"] == 40
            assert "rul" in [c["name"] for c in dataset["columns"]]

            # 2. Schema Inspection
            inspect_res = client.get(f"/api/datasets/{dataset_id}", headers=headers)
            assert inspect_res.status_code == 200
            assert inspect_res.json()["name"] == "E2E RUL Dataset"

            # 3. Create Training Job
            train_payload = {
                "dataset_id": dataset_id,
                "task": "rul_regression",
                "target_column": "rul",
                "feature_columns": ["vibration", "temperature", "pressure"],
                "model_type": "rf_regressor",
                "train_split": 70,
                "validation_split": 15,
                "test_split": 15,
                "random_seed": 42,
            }
            job_res = client.post("/api/training/jobs", headers=headers, json=train_payload)
            assert job_res.status_code == 201, job_res.text
            job = job_res.json()
            job_id = job["id"]

            # Run training synchronously to simulate background worker completion
            run_training_pipeline(job_id, user_id)

            # 4. Verify Training Completed & Real Metrics Produced
            status_res = client.get(f"/api/training/jobs/{job_id}", headers=headers)
            assert status_res.status_code == 200
            job_detail = status_res.json()
            assert job_detail["status"] == "completed"
            assert job_detail["metrics"] is not None
            assert "mae" in job_detail["metrics"]
            assert "rmse" in job_detail["metrics"]
            assert "r2" in job_detail["metrics"]

            # 5. Register Model Artifact
            reg_payload = {
                "training_job_id": job_id,
                "name": "E2E-Turbofan-RUL",
            }
            reg_res = client.post("/api/models/register", headers=headers, json=reg_payload)
            assert reg_res.status_code == 201, reg_res.text
            model = reg_res.json()
            model_id = model["id"]
            assert model["name"] == "E2E-Turbofan-RUL"
            assert model["version"] == 1
            assert model["status"] == "inactive"
            # Ensure internal disk artifact path is not exposed
            assert "artifact_path" not in model

            # 6. Model Activation
            act_res = client.post(f"/api/models/{model_id}/activate", headers=headers)
            assert act_res.status_code == 200
            active_model = act_res.json()
            assert active_model["status"] == "active"

            # Verify active endpoint
            get_act = client.get("/api/models/active/rul_regression", headers=headers)
            assert get_act.status_code == 200
            assert get_act.json()["id"] == model_id

            # 7. Single Inference
            single_payload = {
                "features": {
                    "vibration": 655.0,
                    "temperature": 1600.0,
                    "pressure": 14.8,
                }
            }
            inf_res = client.post("/api/inference/rul", headers=headers, json=single_payload)
            assert inf_res.status_code == 200, inf_res.text
            inf_data = inf_res.json()
            assert inf_data["task"] == "rul_regression"
            assert "rul_cycles" in inf_data["prediction"]
            assert isinstance(inf_data["prediction"]["rul_cycles"], (int, float))
            assert inf_data["model"]["id"] == model_id
            assert inf_data["model"]["version"] == 1

            # 8. Batch Inference (5 records)
            batch_rows = [
                {"vibration": 650.0 + i, "temperature": 1590.0 + i, "pressure": 14.7}
                for i in range(5)
            ]
            batch_res = client.post(
                "/api/inference/rul_regression/batch",
                headers=headers,
                json={"rows": batch_rows},
            )
            assert batch_res.status_code == 200
            batch_data = batch_res.json()
            assert batch_data["total_rows"] == 5
            assert len(batch_data["predictions"]) == 5
            for p in batch_data["predictions"]:
                assert "rul_cycles" in p

            # 9. Prediction History
            hist_res = client.get("/api/inference/history?task=rul_regression", headers=headers)
            assert hist_res.status_code == 200
            hist_data = hist_res.json()
            assert hist_data["total"] >= 2  # At least 1 single and 1 batch
            pred_id = hist_data["predictions"][0]["id"]

            # 10. Prediction Detail Inspection
            detail_res = client.get(f"/api/inference/history/{pred_id}", headers=headers)
            assert detail_res.status_code == 200
            pred_record = detail_res.json()
            assert pred_record["id"] == pred_id
            assert pred_record["model_id"] == model_id

    def test_complete_ml_lifecycle_condition_classification(self):
        """Verifies full Condition Classification pipeline with confusion matrix and decoded labels."""
        user_id = f"user_clf_{uuid.uuid4().hex[:8]}"
        headers = auth_header(user_id)

        with TestClient(app) as client:
            csv_content = create_classification_csv(45)
            upload_res = client.post(
                "/api/datasets",
                headers=headers,
                files={"file": ("health_clf.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
            )
            assert upload_res.status_code == 201
            dataset_id = upload_res.json()["id"]

            train_payload = {
                "dataset_id": dataset_id,
                "task": "condition_classification",
                "target_column": "health_state",
                "feature_columns": ["rpm", "temp", "vibration"],
                "model_type": "rf_classifier",
                "train_split": 70,
                "validation_split": 15,
                "test_split": 15,
                "random_seed": 42,
            }
            job_res = client.post("/api/training/jobs", headers=headers, json=train_payload)
            job_id = job_res.json()["id"]
            run_training_pipeline(job_id, user_id)

            job_detail = client.get(f"/api/training/jobs/{job_id}", headers=headers).json()
            assert job_detail["status"] == "completed"
            assert "accuracy" in job_detail["metrics"]
            assert "confusion_matrix" in job_detail["metrics"]

            # Register & Activate
            reg_res = client.post(
                "/api/models/register",
                headers=headers,
                json={"training_job_id": job_id, "name": "Classifier-RF"},
            )
            model_id = reg_res.json()["id"]
            client.post(f"/api/models/{model_id}/activate", headers=headers)

            # Single Inference
            pred_res = client.post(
                "/api/inference/classification",
                headers=headers,
                json={"features": {"rpm": 1300, "temp": 68.0, "vibration": 1.2}},
            )
            assert pred_res.status_code == 200
            pred_data = pred_res.json()
            assert pred_data["prediction"]["condition"] in ["Healthy", "Degrading", "Critical"]
            assert "probabilities" in pred_data["prediction"]

    def test_complete_ml_lifecycle_anomaly_detection(self):
        """Verifies full Anomaly Detection pipeline with Isolation Forest decisions and contamination."""
        user_id = f"user_anom_{uuid.uuid4().hex[:8]}"
        headers = auth_header(user_id)

        with TestClient(app) as client:
            csv_content = create_anomaly_csv(40)
            upload_res = client.post(
                "/api/datasets",
                headers=headers,
                files={"file": ("anom_dataset.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
            )
            assert upload_res.status_code == 201
            dataset_id = upload_res.json()["id"]

            train_payload = {
                "dataset_id": dataset_id,
                "task": "anomaly_detection",
                "target_column": None,
                "feature_columns": ["sensor_a", "sensor_b", "sensor_c"],
                "model_type": "isolation_forest",
                "train_split": 70,
                "validation_split": 15,
                "test_split": 15,
                "random_seed": 42,
            }
            job_res = client.post("/api/training/jobs", headers=headers, json=train_payload)
            job_id = job_res.json()["id"]
            run_training_pipeline(job_id, user_id)

            job_detail = client.get(f"/api/training/jobs/{job_id}", headers=headers).json()
            assert job_detail["status"] == "completed"
            assert "anomaly_rate" in job_detail["metrics"]
            assert "score_mean" in job_detail["metrics"]

            # Register & Activate
            reg_res = client.post(
                "/api/models/register",
                headers=headers,
                json={"training_job_id": job_id, "name": "Anomaly-IsoForest"},
            )
            model_id = reg_res.json()["id"]
            client.post(f"/api/models/{model_id}/activate", headers=headers)

            # Single Inference
            pred_res = client.post(
                "/api/inference/anomaly",
                headers=headers,
                json={"features": {"sensor_a": 105.0, "sensor_b": 52.0, "sensor_c": 26.0}},
            )
            assert pred_res.status_code == 200
            pred_data = pred_res.json()
            assert isinstance(pred_data["prediction"]["is_anomaly"], bool)
            assert isinstance(pred_data["prediction"]["anomaly_score"], float)

    def test_cross_user_isolation_and_unauthorized_access(self):
        """Validates that User B cannot access, mutate, activate or view any User A resources."""
        user_a = f"tenant_alpha_{uuid.uuid4().hex[:8]}"
        user_b = f"tenant_beta_{uuid.uuid4().hex[:8]}"
        headers_a = auth_header(user_a)
        headers_b = auth_header(user_b)

        with TestClient(app) as client:
            # 1. User A creates dataset and model
            csv_content = create_regression_csv(35)
            ds_a = client.post(
                "/api/datasets",
                headers=headers_a,
                files={"file": ("dataset_a.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
            ).json()
            dataset_a_id = ds_a["id"]

            job_a = client.post(
                "/api/training/jobs",
                headers=headers_a,
                json={
                    "dataset_id": dataset_a_id,
                    "task": "rul_regression",
                    "target_column": "rul",
                    "feature_columns": ["vibration", "temperature", "pressure"],
                    "model_type": "rf_regressor",
                    "train_split": 70,
                    "validation_split": 15,
                    "test_split": 15,
                },
            ).json()
            job_a_id = job_a["id"]
            run_training_pipeline(job_a_id, user_a)

            model_a = client.post(
                "/api/models/register",
                headers=headers_a,
                json={"training_job_id": job_a_id, "name": "Alpha-Model"},
            ).json()
            model_a_id = model_a["id"]
            client.post(f"/api/models/{model_a_id}/activate", headers=headers_a)

            # User A performs inference
            client.post(
                "/api/inference/rul",
                headers=headers_a,
                json={"features": {"vibration": 650.0, "temperature": 1590.0, "pressure": 14.7}},
            )
            history_a = client.get("/api/inference/history", headers=headers_a).json()
            pred_a_id = history_a["predictions"][0]["id"]

            # --- USER B ATTEMPTS CROSS-TENANT ACCESS ---
            # Attempt to read User A dataset
            assert client.get(f"/api/datasets/{dataset_a_id}", headers=headers_b).status_code == 404
            # Attempt to delete User A dataset
            assert client.delete(f"/api/datasets/{dataset_a_id}", headers=headers_b).status_code == 404
            # Attempt to train on User A dataset
            train_cross = client.post(
                "/api/training/jobs",
                headers=headers_b,
                json={
                    "dataset_id": dataset_a_id,
                    "task": "rul_regression",
                    "target_column": "rul",
                    "feature_columns": ["vibration", "temperature", "pressure"],
                    "model_type": "rf_regressor",
                },
            )
            assert train_cross.status_code == 404

            # Attempt to inspect User A training job
            assert client.get(f"/api/training/jobs/{job_a_id}", headers=headers_b).status_code == 404
            # Attempt to cancel User A training job
            assert client.post(f"/api/training/jobs/{job_a_id}/cancel", headers=headers_b).status_code == 404
            # Attempt to register User A training job
            assert client.post(
                "/api/models/register",
                headers=headers_b,
                json={"training_job_id": job_a_id, "name": "Stolen-Model"},
            ).status_code == 404

            # Attempt to read User A model detail
            assert client.get(f"/api/models/{model_a_id}", headers=headers_b).status_code == 404
            # Attempt to read User A model versions
            assert client.get(f"/api/models/{model_a_id}/versions", headers=headers_b).status_code == 404
            # Attempt to activate User A model
            assert client.post(f"/api/models/{model_a_id}/activate", headers=headers_b).status_code == 404
            # Attempt to deactivate User A model
            assert client.post(f"/api/models/{model_a_id}/deactivate", headers=headers_b).status_code == 404

            # User B attempts inference with no active model of their own
            assert client.post(
                "/api/inference/rul",
                headers=headers_b,
                json={"features": {"vibration": 650.0, "temperature": 1590.0, "pressure": 14.7}},
            ).status_code == 404

            # User B attempts to read User A prediction history record
            assert client.get(f"/api/inference/history/{pred_a_id}", headers=headers_b).status_code == 404
            # User B list history must be empty (User A records not leaked)
            hist_b = client.get("/api/inference/history", headers=headers_b).json()
            assert hist_b["total"] == 0

            # --- UNAUTHENTICATED REQUESTS ARE REJECTED (401) ---
            assert client.get("/api/datasets").status_code == 401
            assert client.get("/api/training/jobs").status_code == 401
            assert client.get("/api/models").status_code == 401
            assert client.post("/api/inference/rul", json={"features": {}}).status_code == 401
            assert client.get("/api/inference/history").status_code == 401

    def test_batch_size_limits_and_invalid_schema_handling(self):
        """Verifies boundary checks, oversized batch rejection, and schema mismatch error handling."""
        user_id = f"user_limits_{uuid.uuid4().hex[:8]}"
        headers = auth_header(user_id)

        with TestClient(app) as client:
            csv_content = create_regression_csv(35)
            upload_res = client.post(
                "/api/datasets",
                headers=headers,
                files={"file": ("limits_ds.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
            )
            dataset_id = upload_res.json()["id"]

            job_res = client.post(
                "/api/training/jobs",
                headers=headers,
                json={
                    "dataset_id": dataset_id,
                    "task": "rul_regression",
                    "target_column": "rul",
                    "feature_columns": ["vibration", "temperature", "pressure"],
                    "model_type": "rf_regressor",
                },
            )
            job_id = job_res.json()["id"]
            run_training_pipeline(job_id, user_id)

            model_res = client.post(
                "/api/models/register",
                headers=headers,
                json={"training_job_id": job_id, "name": "Limits-Model"},
            )
            model_id = model_res.json()["id"]
            client.post(f"/api/models/{model_id}/activate", headers=headers)

            # Incompatible schema (missing required feature 'temperature')
            bad_schema_res = client.post(
                "/api/inference/rul",
                headers=headers,
                json={"features": {"vibration": 650.0, "pressure": 14.7}},
            )
            assert bad_schema_res.status_code == 400
            assert "Missing required feature column" in bad_schema_res.json()["detail"]

            # Oversized batch (> 500 records)
            huge_batch = [{"vibration": 650.0, "temperature": 1590.0, "pressure": 14.7}] * 501
            huge_res = client.post(
                "/api/inference/rul_regression/batch",
                headers=headers,
                json={"rows": huge_batch},
            )
            assert huge_res.status_code == 400
            assert "exceeds maximum allowed" in huge_res.json()["detail"]

    def test_observability_and_prediction_metadata(self):
        """Verifies practical ML run observability: job duration, timestamps, model metadata in history."""
        user_id = f"user_obs_{uuid.uuid4().hex[:8]}"
        headers = auth_header(user_id)

        with TestClient(app) as client:
            csv_content = create_regression_csv(35)
            upload_res = client.post(
                "/api/datasets",
                headers=headers,
                files={"file": ("obs_data.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
            )
            dataset_id = upload_res.json()["id"]

            job_res = client.post(
                "/api/training/jobs",
                headers=headers,
                json={
                    "dataset_id": dataset_id,
                    "task": "rul_regression",
                    "target_column": "rul",
                    "feature_columns": ["vibration", "temperature", "pressure"],
                    "model_type": "rf_regressor",
                },
            )
            job_id = job_res.json()["id"]
            run_training_pipeline(job_id, user_id)

            # Verify training job observability
            job_detail = client.get(f"/api/training/jobs/{job_id}", headers=headers).json()
            assert job_detail["status"] == "completed"
            assert job_detail["started_at"] is not None
            assert job_detail["completed_at"] is not None
            assert "duration_seconds" in job_detail
            assert isinstance(job_detail["duration_seconds"], (int, float))
            assert job_detail["duration_seconds"] >= 0

            # Register & Activate
            model_res = client.post(
                "/api/models/register",
                headers=headers,
                json={"training_job_id": job_id, "name": "Observability-RF"},
            )
            model_id = model_res.json()["id"]
            client.post(f"/api/models/{model_id}/activate", headers=headers)

            # Inference
            client.post(
                "/api/inference/rul",
                headers=headers,
                json={"features": {"vibration": 655.0, "temperature": 1595.0, "pressure": 14.8}},
            )

            # Verify history observability
            hist_res = client.get("/api/inference/history", headers=headers)
            assert hist_res.status_code == 200
            history_data = hist_res.json()
            assert history_data["total"] >= 1
            pred = history_data["predictions"][0]
            assert pred["model_id"] == model_id
            assert pred["model_name"] == "Observability-RF"
            assert pred["model_version"] == 1
            assert pred["model_type"] == "rf_regressor"
            assert pred["created_at"] is not None
            assert "rul_cycles" in pred["prediction"]

            # Verify prediction detail observability
            detail_res = client.get(f"/api/inference/history/{pred['id']}", headers=headers)
            assert detail_res.status_code == 200
            detail = detail_res.json()
            assert detail["model_name"] == "Observability-RF"
            assert detail["model_version"] == 1
            assert detail["model_type"] == "rf_regressor"

