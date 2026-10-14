"""Phase 7 Backend Tests: Model Comparison & Evaluation History Service."""
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
from app.db.models import ModelRegistryRecord, User

settings.ENVIRONMENT = "test"


def auth_header(user_id: str):
    return {"Authorization": f"Bearer test_token_{user_id}"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_unauthenticated_comparison_rejected(client):
    res = client.post("/api/models/compare", json={"model_ids": ["benchmark_rf", "benchmark_lstm"]})
    assert res.status_code == 401


def test_comparison_requires_at_least_two_models(client):
    res = client.post(
        "/api/models/compare",
        headers=auth_header("user_comp_1"),
        json={"model_ids": ["benchmark_rf"]},
    )
    assert res.status_code == 400
    assert "At least 2 model IDs" in res.json()["detail"]


def test_compare_compatible_benchmark_models(client):
    res = client.post(
        "/api/models/compare",
        headers=auth_header("user_comp_1"),
        json={"model_ids": ["benchmark_rf", "benchmark_lstm"]},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_compatible"] is True
    assert data["task"] == "rul_regression"
    assert data["identical_evaluation_dataset"] is True
    assert len(data["models"]) == 2
    assert "metric_comparison" in data
    # Check that MAE, RMSE, R2 are present
    metric_keys = [m["metric_key"] for m in data["metric_comparison"]]
    assert "mae" in metric_keys
    assert "rmse" in metric_keys
    assert "r2" in metric_keys

    # Check verified benchmark metric values
    mae_item = next(m for m in data["metric_comparison"] if m["metric_key"] == "mae")
    assert mae_item["values"]["benchmark_rf"] == 13.2202
    assert mae_item["values"]["benchmark_lstm"] == 11.3005
    # Best model for MAE (lower is better): LSTM
    assert mae_item["best_model_id"] == "benchmark_lstm"


def test_compare_incompatible_tasks_rejected_scientifically(client):
    res = client.post(
        "/api/models/compare",
        headers=auth_header("user_comp_1"),
        json={"model_ids": ["benchmark_rf", "benchmark_isolation_forest"]},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_compatible"] is False
    assert "incompatibility_reason" in data
    assert "different ML tasks" in data["incompatibility_reason"]
    assert "regression" in data["incompatibility_reason"].lower()


def test_compare_nonexistent_model_returns_404(client):
    res = client.post(
        "/api/models/compare",
        headers=auth_header("user_comp_1"),
        json={"model_ids": ["benchmark_rf", "model_does_not_exist_999"]},
    )
    assert res.status_code == 404


def test_model_evaluation_history(client):
    uid = f"comp_hist_{uuid.uuid4().hex[:6]}"
    model_name = f"turbofan_regressor_{uuid.uuid4().hex[:4]}"

    db = next(get_db())
    try:
        # Ensure user exists
        u = db.query(User).filter(User.id == uid).first()
        if not u:
            u = User(id=uid, email=f"{uid}@test.com", first_name="Hist", last_name="User")
            db.add(u)
            db.commit()

        # Add 2 training jobs
        from app.db.models import TrainingJobRecord
        ds_id = f"ds_{uuid.uuid4().hex[:6]}"
        j1 = TrainingJobRecord(
            id=f"job_v1_{uuid.uuid4().hex[:6]}",
            user_id=uid,
            dataset_id=ds_id,
            task="rul_regression",
            model_type="rf_regressor",
            status="completed",
        )
        j2 = TrainingJobRecord(
            id=f"job_v2_{uuid.uuid4().hex[:6]}",
            user_id=uid,
            dataset_id=ds_id,
            task="rul_regression",
            model_type="rf_regressor",
            status="completed",
        )
        db.add_all([j1, j2])
        db.commit()

        # Add 2 versions
        r1 = ModelRegistryRecord(
            id=f"mod_v1_{uuid.uuid4().hex[:6]}",
            user_id=uid,
            training_job_id=j1.id,
            dataset_id=ds_id,
            name=model_name,
            version=1,
            task="rul_regression",
            model_type="rf_regressor",
            status="archived",
            metrics_json=json.dumps({"mae": 14.5, "rmse": 19.2, "r2": 0.78}),
            artifact_path="dummy/path/v1.joblib",
        )
        r2 = ModelRegistryRecord(
            id=f"mod_v2_{uuid.uuid4().hex[:6]}",
            user_id=uid,
            training_job_id=j2.id,
            dataset_id=ds_id,
            name=model_name,
            version=2,
            task="rul_regression",
            model_type="rf_regressor",
            status="active",
            metrics_json=json.dumps({"mae": 12.1, "rmse": 16.0, "r2": 0.84}),
            artifact_path="dummy/path/v2.joblib",
        )
        db.add_all([r1, r2])
        db.commit()

        # Fetch evaluation history
        res = client.get(f"/api/models/{model_name}/evaluation-history", headers=auth_header(uid))
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == model_name
        assert data["total_versions"] == 2
        assert len(data["versions"]) == 2
        assert data["versions"][0]["version"] == 1
        assert data["versions"][0]["metrics"]["mae"] == 14.5
        assert data["versions"][1]["version"] == 2
        assert data["versions"][1]["metrics"]["mae"] == 12.1

        # Another user cannot view this model's history
        res_other = client.get(f"/api/models/{model_name}/evaluation-history", headers=auth_header("other_user"))
        assert res_other.status_code == 404

    finally:
        db.close()
