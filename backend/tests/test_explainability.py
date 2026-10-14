"""Phase 7 Backend Tests: ML Explainability Service & Endpoints."""
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
from app.db.models import ModelPredictionRecord, ModelRegistryRecord, User
from app.ml_service import FEATURE_COLS

settings.ENVIRONMENT = "test"


def auth_header(user_id: str):
    return {"Authorization": f"Bearer test_token_{user_id}"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_unauthenticated_explain_rejected(client):
    res = client.get("/api/models/benchmark_rf/explain")
    assert res.status_code == 401


def test_benchmark_rf_global_explain(client):
    res = client.get("/api/models/benchmark_rf/explain", headers=auth_header("user_p7_1"))
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "benchmark_rf"
    assert data["is_benchmark"] is True
    assert data["supported"] is True
    assert data["explanation_method"] == "mean_decrease_impurity"
    assert "features" in data
    assert len(data["features"]) > 0
    # Verify rankings and structure
    first_feat = data["features"][0]
    assert first_feat["rank"] == 1
    assert "importance" in first_feat
    assert "label" in first_feat
    assert "unit" in first_feat
    assert "limitations" in data
    assert "causation" in data["limitations"].lower()


def test_benchmark_lstm_global_explain(client):
    res = client.get("/api/models/benchmark_lstm/explain", headers=auth_header("user_p7_1"))
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "benchmark_lstm"
    assert data["is_benchmark"] is True
    assert data["supported"] is True
    assert data["explanation_method"] == "sequential_channel_perturbation_sensitivity"
    assert len(data["features"]) > 0
    assert "sensitivity" in data["limitations"].lower()


def test_benchmark_isolation_forest_global_explain(client):
    res = client.get("/api/models/benchmark_isolation_forest/explain", headers=auth_header("user_p7_1"))
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "benchmark_isolation_forest"
    assert data["is_benchmark"] is True
    assert data["supported"] is True
    assert data["explanation_method"] == "feature_perturbation_sensitivity"
    assert len(data["features"]) > 0
    assert "unsupervised" in data["limitations"].lower()


def test_benchmark_unknown_returns_404(client):
    res = client.get("/api/models/benchmark/non_existent_model/explain", headers=auth_header("user_p7_1"))
    assert res.status_code == 404


def test_benchmark_rf_local_explain_payload(client):
    # Construct a valid feature payload for benchmark
    features = {col: 100.0 for col in FEATURE_COLS}
    res = client.post(
        "/api/models/benchmark_rf/explain/local",
        headers=auth_header("user_p7_1"),
        json={"features": features},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "benchmark_rf"
    assert "contributions" in data
    assert len(data["contributions"]) == len(FEATURE_COLS)
    assert "contribution" in data["contributions"][0]
    assert "impact_direction" in data["contributions"][0]


def test_benchmark_local_explain_missing_features(client):
    res = client.post(
        "/api/models/benchmark_rf/explain/local",
        headers=auth_header("user_p7_1"),
        json={"features": {"s2": 100.0}},
    )
    assert res.status_code == 400
    assert "Missing required benchmark feature columns" in res.json()["detail"]


def test_prediction_history_local_explain_isolation(client):
    uid_a = f"p7_user_a_{uuid.uuid4().hex[:6]}"
    uid_b = f"p7_user_b_{uuid.uuid4().hex[:6]}"

    # Inject a prediction record for user A
    db = next(get_db())
    try:
        # Ensure user A exists
        user_a = db.query(User).filter(User.id == uid_a).first()
        if not user_a:
            user_a = User(id=uid_a, email=f"{uid_a}@test.com", first_name="User", last_name="A")
            db.add(user_a)
            db.commit()

        # Create TrainingJobRecord first
        job_id = f"job_p7_{uuid.uuid4().hex[:6]}"
        from app.db.models import TrainingJobRecord
        ds_id = f"ds_{uuid.uuid4().hex[:6]}"
        job = TrainingJobRecord(
            id=job_id,
            user_id=uid_a,
            dataset_id=ds_id,
            task="rul_regression",
            model_type="rf_regressor",
            status="completed",
            feature_columns_json=json.dumps(FEATURE_COLS),
        )
        db.add(job)
        db.commit()

        reg_b = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == "benchmark_rf").first()
        if not reg_b:
            reg_b = ModelRegistryRecord(
                id="benchmark_rf",
                user_id=uid_a,
                training_job_id=job_id,
                dataset_id=ds_id,
                name="NASA C-MAPSS Random Forest Regressor",
                version=1,
                task="rul_regression",
                model_type="rf_regressor",
                status="active_benchmark",
                artifact_path="artifacts/rul/rf_baseline.joblib",
            )
            db.add(reg_b)
            db.commit()

        pred_id = f"pred_hist_{uuid.uuid4().hex[:8]}"
        features = {col: 120.0 for col in FEATURE_COLS}
        pred_record = ModelPredictionRecord(
            id=pred_id,
            user_id=uid_a,
            model_id="benchmark_rf",
            task="rul_regression",
            input_metadata_json=json.dumps({"features": features}),
            prediction_json=json.dumps({"rul_cycles": 88.5}),
        )
        db.add(pred_record)
        db.commit()

        # User A can explain their historical prediction
        res_a = client.get(f"/api/inference/history/{pred_id}/explain", headers=auth_header(uid_a))
        assert res_a.status_code == 200
        data_a = res_a.json()
        assert data_a["prediction_id"] == pred_id
        assert len(data_a["contributions"]) > 0

        # User B cannot access User A's prediction explanation (404)
        res_b = client.get(f"/api/inference/history/{pred_id}/explain", headers=auth_header(uid_b))
        assert res_b.status_code == 404

    finally:
        db.close()
