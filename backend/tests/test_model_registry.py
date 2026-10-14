"""Comprehensive test suite for Backend Part 3: Model Registry, Versioning & Model Activation.

Tests cover:
1. Register completed RF RUL training job
2. Register completed LSTM RUL training job
3. Register completed classification job
4. Register completed anomaly job
5. Reject incomplete training job (400)
6. Reject missing artifact (400)
7. Reject duplicate registration (409)
8. Automatic version 1 creation
9. Automatic version increment (v1 -> v2)
10. List user's models
11. Filter models by task
12. Get model detail
13. Get version history
14. Activate model
15. Deactivate previously active model when another same-task model activates
16. Activate model (idempotent)
17. Deactivate model
18. Get active model
19. No active model returns 404
20. Cross-user model access returns 404
21. Cross-user activation returns 404
22. Cross-user deactivation returns 404
23. Invalid task returns 400
24. Artifact path traversal protection
25. Database persistence after registration
"""
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
from app.db.models import TrainingJobRecord, ModelRegistryRecord

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


def create_completed_job(
    client: TestClient,
    user: str,
    task: str = "rul_regression",
    model_type: str = "rf_regressor",
    target_col: str = "rul",
) -> str:
    if task == "condition_classification":
        csv_data = make_classification_csv(36)
        target = "condition"
        features = ["cycle", "vibration", "temperature", "pressure"]
    elif task == "anomaly_detection":
        csv_data = make_anomaly_csv(40)
        target = None
        features = ["vibration", "temperature", "pressure"]
    else:
        csv_data = make_regression_csv(35)
        target = target_col
        features = ["cycle", "vibration", "temperature", "pressure"]

    ds_id = upload_dataset(client, user, csv_data, f"DS_{uuid.uuid4().hex[:6]}")
    payload = {
        "dataset_id": ds_id,
        "task": task,
        "model_type": model_type,
        "target_column": target,
        "feature_columns": features,
        "train_split": 70,
        "validation_split": 15,
        "test_split": 15,
        "random_seed": 42,
    }
    resp = client.post("/api/training/jobs", headers=auth_header(user), json=payload)
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["id"]

    # Verify completed
    get_resp = client.get(f"/api/training/jobs/{job_id}", headers=auth_header(user))
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "completed"
    return job_id


# ---------------------------------------------------------------------------
# 1. Register completed RF RUL training job
# ---------------------------------------------------------------------------
def test_register_completed_rf_rul_training_job():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user, "rul_regression", "rf_regressor")

        reg_payload = {"training_job_id": job_id, "name": "Production RUL Random Forest"}
        resp = client.post("/api/models/register", headers=auth_header(user), json=reg_payload)
        assert resp.status_code == 201, resp.text
        data = resp.json()

        assert data["id"].startswith("model_")
        assert data["name"] == "Production RUL Random Forest"
        assert data["task"] == "rul_regression"
        assert data["model_type"] == "rf_regressor"
        assert data["version"] == 1
        assert data["status"] == "inactive"
        assert data["training_job_id"] == job_id
        assert data["metrics"] is not None
        assert "mae" in data["metrics"]
        assert "rmse" in data["metrics"]
        assert "r2" in data["metrics"]
        assert data["activated_at"] is None
        assert data["deactivated_at"] is None


# ---------------------------------------------------------------------------
# 2. Register completed LSTM RUL training job
# ---------------------------------------------------------------------------
def test_register_completed_lstm_rul_training_job():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user, "rul_regression", "lstm_regressor")

        reg_payload = {"training_job_id": job_id, "name": "Deep LSTM RUL"}
        resp = client.post("/api/models/register", headers=auth_header(user), json=reg_payload)
        assert resp.status_code == 201, resp.text
        data = resp.json()

        assert data["name"] == "Deep LSTM RUL"
        assert data["model_type"] == "lstm_regressor"
        assert data["version"] == 1
        assert "mae" in data["metrics"]


# ---------------------------------------------------------------------------
# 3. Register completed classification job
# ---------------------------------------------------------------------------
def test_register_completed_classification_job():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user, "condition_classification", "rf_classifier")

        reg_payload = {"training_job_id": job_id, "name": "Health Classifier"}
        resp = client.post("/api/models/register", headers=auth_header(user), json=reg_payload)
        assert resp.status_code == 201, resp.text
        data = resp.json()

        assert data["task"] == "condition_classification"
        assert data["model_type"] == "rf_classifier"
        assert "accuracy" in data["metrics"]
        assert "confusion_matrix" in data["metrics"]


# ---------------------------------------------------------------------------
# 4. Register completed anomaly job
# ---------------------------------------------------------------------------
def test_register_completed_anomaly_job():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user, "anomaly_detection", "isolation_forest")

        reg_payload = {"training_job_id": job_id, "name": "Telemetry Anomaly Detector"}
        resp = client.post("/api/models/register", headers=auth_header(user), json=reg_payload)
        assert resp.status_code == 201, resp.text
        data = resp.json()

        assert data["task"] == "anomaly_detection"
        assert data["model_type"] == "isolation_forest"
        assert "anomaly_count" in data["metrics"]


# ---------------------------------------------------------------------------
# 5. Reject incomplete training job (400)
# ---------------------------------------------------------------------------
def test_reject_incomplete_training_job():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(30), "Incomplete DS")

        # Create a job record with status="training"
        job_id = f"job_inc_{uuid.uuid4().hex[:8]}"
        db = next(get_db())
        try:
            job = TrainingJobRecord(
                id=job_id,
                user_id=user,
                dataset_id=ds_id,
                task="rul_regression",
                model_type="rf_regressor",
                status="training",
                current_step="Training in progress",
            )
            db.add(job)
            db.commit()
        finally:
            db.close()

        resp = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job_id, "name": "Premature Model"},
        )
        assert resp.status_code == 400
        assert "must be 'completed'" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 6. Reject missing artifact (400)
# ---------------------------------------------------------------------------
def test_reject_missing_artifact():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user)

        # Remove the artifact from disk
        db = next(get_db())
        try:
            job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
            if job and job.artifact_path and os.path.exists(job.artifact_path):
                os.remove(job.artifact_path)
        finally:
            db.close()

        resp = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job_id, "name": "Missing Artifact Model"},
        )
        assert resp.status_code == 400
        assert "does not exist on disk" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 7. Reject duplicate registration (409)
# ---------------------------------------------------------------------------
def test_reject_duplicate_registration():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user)

        resp1 = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job_id, "name": "Unique Model"},
        )
        assert resp1.status_code == 201

        resp2 = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job_id, "name": "Duplicate Registration"},
        )
        assert resp2.status_code == 409
        assert "already registered" in resp2.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 8. Automatic version 1 creation
# ---------------------------------------------------------------------------
def test_automatic_version_1_creation():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user)

        resp = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job_id, "name": "Versioned Engine"},
        )
        assert resp.status_code == 201
        assert resp.json()["version"] == 1


# ---------------------------------------------------------------------------
# 9. Automatic version increment (v1 -> v2)
# ---------------------------------------------------------------------------
def test_automatic_version_increment():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job1 = create_completed_job(client, user, "rul_regression", "rf_regressor")
        job2 = create_completed_job(client, user, "rul_regression", "rf_regressor")

        resp1 = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job1, "name": "Fleet Model"},
        )
        assert resp1.status_code == 201
        assert resp1.json()["version"] == 1

        resp2 = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job2, "name": "Fleet Model"},
        )
        assert resp2.status_code == 201
        assert resp2.json()["version"] == 2


# ---------------------------------------------------------------------------
# 10. List user's models
# ---------------------------------------------------------------------------
def test_list_users_models():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job1 = create_completed_job(client, user, "rul_regression")
        job2 = create_completed_job(client, user, "condition_classification", "rf_classifier")

        client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job1, "name": "M1"})
        client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job2, "name": "M2"})

        resp = client.get("/api/models", headers=auth_header(user))
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) >= 2
        model_names = [m["name"] for m in data["models"]]
        assert "M1" in model_names
        assert "M2" in model_names


# ---------------------------------------------------------------------------
# 11. Filter models by task
# ---------------------------------------------------------------------------
def test_filter_models_by_task():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_rul = create_completed_job(client, user, "rul_regression")
        job_cls = create_completed_job(client, user, "condition_classification", "rf_classifier")

        client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job_rul, "name": "RUL Filter Test"})
        client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job_cls, "name": "CLS Filter Test"})

        resp = client.get("/api/models?task=rul_regression", headers=auth_header(user))
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert all(m["task"] == "rul_regression" for m in models)
        assert any(m["name"] == "RUL Filter Test" for m in models)
        assert not any(m["name"] == "CLS Filter Test" for m in models)


# ---------------------------------------------------------------------------
# 12. Get model detail
# ---------------------------------------------------------------------------
def test_get_model_detail():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user)

        reg_resp = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job_id, "name": "Detail Model"},
        )
        model_id = reg_resp.json()["id"]

        resp = client.get(f"/api/models/{model_id}", headers=auth_header(user))
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["id"] == model_id
        assert detail["name"] == "Detail Model"
        assert "training_metadata" in detail
        assert "metrics" in detail
        assert detail["status"] == "inactive"


# ---------------------------------------------------------------------------
# 13. Get version history
# ---------------------------------------------------------------------------
def test_get_version_history():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job1 = create_completed_job(client, user)
        job2 = create_completed_job(client, user)

        r1 = client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job1, "name": "History Model"})
        r2 = client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job2, "name": "History Model"})
        model1_id = r1.json()["id"]

        resp = client.get(f"/api/models/{model1_id}/versions", headers=auth_header(user))
        assert resp.status_code == 200
        versions = resp.json()
        assert len(versions) == 2
        # Descending sort: v2 first, then v1
        assert versions[0]["version"] == 2
        assert versions[1]["version"] == 1


# ---------------------------------------------------------------------------
# 14. Activate model
# ---------------------------------------------------------------------------
def test_activate_model():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user)

        reg_resp = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job_id, "name": "To Activate"},
        )
        model_id = reg_resp.json()["id"]

        act_resp = client.post(f"/api/models/{model_id}/activate", headers=auth_header(user))
        assert act_resp.status_code == 200
        act_data = act_resp.json()
        assert act_data["status"] == "active"
        assert act_data["activated_at"] is not None
        assert act_data["deactivated_at"] is None


# ---------------------------------------------------------------------------
# 15. Deactivate previously active model when another same-task model activates
# ---------------------------------------------------------------------------
def test_deactivate_previous_active_model_when_new_activates():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job1 = create_completed_job(client, user, "rul_regression")
        job2 = create_completed_job(client, user, "rul_regression")

        m1 = client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job1, "name": "Model 1"}).json()["id"]
        m2 = client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job2, "name": "Model 2"}).json()["id"]

        # Activate model 1
        client.post(f"/api/models/{m1}/activate", headers=auth_header(user))
        det1_active = client.get(f"/api/models/{m1}", headers=auth_header(user)).json()
        assert det1_active["status"] == "active"

        # Activate model 2 -> model 1 must become inactive
        client.post(f"/api/models/{m2}/activate", headers=auth_header(user))

        det1_inactive = client.get(f"/api/models/{m1}", headers=auth_header(user)).json()
        assert det1_inactive["status"] == "inactive"
        assert det1_inactive["deactivated_at"] is not None

        det2_active = client.get(f"/api/models/{m2}", headers=auth_header(user)).json()
        assert det2_active["status"] == "active"


# ---------------------------------------------------------------------------
# 16. Activate model (idempotent)
# ---------------------------------------------------------------------------
def test_activate_model_idempotent():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user)

        m = client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job_id, "name": "Idempotent Model"}).json()["id"]

        r1 = client.post(f"/api/models/{m}/activate", headers=auth_header(user))
        assert r1.status_code == 200
        assert r1.json()["status"] == "active"

        # Re-activate
        r2 = client.post(f"/api/models/{m}/activate", headers=auth_header(user))
        assert r2.status_code == 200
        assert r2.json()["status"] == "active"


# ---------------------------------------------------------------------------
# 17. Deactivate model
# ---------------------------------------------------------------------------
def test_deactivate_model():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user)

        m = client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job_id, "name": "Deactivate Target"}).json()["id"]
        client.post(f"/api/models/{m}/activate", headers=auth_header(user))

        deact_resp = client.post(f"/api/models/{m}/deactivate", headers=auth_header(user))
        assert deact_resp.status_code == 200
        data = deact_resp.json()
        assert data["status"] == "inactive"
        assert data["deactivated_at"] is not None

        # Deactivate again (idempotent)
        deact2 = client.post(f"/api/models/{m}/deactivate", headers=auth_header(user))
        assert deact2.status_code == 200
        assert deact2.json()["status"] == "inactive"


# ---------------------------------------------------------------------------
# 18. Get active model
# ---------------------------------------------------------------------------
def test_get_active_model():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user, "rul_regression")

        m = client.post("/api/models/register", headers=auth_header(user), json={"training_job_id": job_id, "name": "Active RUL Model"}).json()["id"]
        client.post(f"/api/models/{m}/activate", headers=auth_header(user))

        resp = client.get("/api/models/active/rul_regression", headers=auth_header(user))
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == m
        assert data["status"] == "active"
        assert data["task"] == "rul_regression"


# ---------------------------------------------------------------------------
# 19. No active model returns 404
# ---------------------------------------------------------------------------
def test_no_active_model_returns_404():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        resp = client.get("/api/models/active/rul_regression", headers=auth_header(user))
        assert resp.status_code == 404
        assert "no active model" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 20. Cross-user model access returns 404
# ---------------------------------------------------------------------------
def test_cross_user_model_access_returns_404():
    with TestClient(app) as client:
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        job_a = create_completed_job(client, user_a)
        m_a = client.post("/api/models/register", headers=auth_header(user_a), json={"training_job_id": job_a, "name": "User A Model"}).json()["id"]

        resp = client.get(f"/api/models/{m_a}", headers=auth_header(user_b))
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 21. Cross-user activation returns 404
# ---------------------------------------------------------------------------
def test_cross_user_activation_returns_404():
    with TestClient(app) as client:
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        job_a = create_completed_job(client, user_a)
        m_a = client.post("/api/models/register", headers=auth_header(user_a), json={"training_job_id": job_a, "name": "User A Model"}).json()["id"]

        resp = client.post(f"/api/models/{m_a}/activate", headers=auth_header(user_b))
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 22. Cross-user deactivation returns 404
# ---------------------------------------------------------------------------
def test_cross_user_deactivation_returns_404():
    with TestClient(app) as client:
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        job_a = create_completed_job(client, user_a)
        m_a = client.post("/api/models/register", headers=auth_header(user_a), json={"training_job_id": job_a, "name": "User A Model"}).json()["id"]

        resp = client.post(f"/api/models/{m_a}/deactivate", headers=auth_header(user_b))
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 23. Invalid task returns 400
# ---------------------------------------------------------------------------
def test_invalid_task_returns_400():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        resp = client.get("/api/models/active/invalid_task_name", headers=auth_header(user))
        assert resp.status_code == 400
        assert "invalid task" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 24. Artifact path traversal protection
# ---------------------------------------------------------------------------
def test_artifact_path_traversal_protection():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        ds_id = upload_dataset(client, user, make_regression_csv(30), "Traversal DS")

        # Create a job with malicious artifact path
        job_id = f"job_trav_{uuid.uuid4().hex[:8]}"
        db = next(get_db())
        try:
            job = TrainingJobRecord(
                id=job_id,
                user_id=user,
                dataset_id=ds_id,
                task="rul_regression",
                model_type="rf_regressor",
                status="completed",
                artifact_path="../../etc/passwd",
                metrics_json=json.dumps({"mae": 1.0}),
            )
            db.add(job)
            db.commit()
        finally:
            db.close()

        resp = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job_id, "name": "Malicious Model"},
        )
        assert resp.status_code == 400
        assert "traversal" in resp.json()["detail"].lower() or "outside" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 25. Database persistence after registration
# ---------------------------------------------------------------------------
def test_database_persistence_after_registration():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        job_id = create_completed_job(client, user)

        resp = client.post(
            "/api/models/register",
            headers=auth_header(user),
            json={"training_job_id": job_id, "name": "Persisted Model"},
        )
        assert resp.status_code == 201
        model_id = resp.json()["id"]

        # Directly inspect database
        db = next(get_db())
        try:
            record = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == model_id).first()
            assert record is not None
            assert record.user_id == user
            assert record.name == "Persisted Model"
            assert record.version == 1
            assert record.status == "inactive"
            assert record.training_job_id == job_id
            assert record.created_at is not None
        finally:
            db.close()
