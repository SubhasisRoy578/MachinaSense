"""Comprehensive test suite for Backend Part 4: Real User-Trained Model Inference.

Tests cover:
1. RUL inference with active RF model
2. RUL inference with active LSTM model
3. Classification inference
4. Anomaly inference
5. Batch RUL inference
6. Batch classification inference
7. Batch anomaly inference
8. No active RUL model -> 404
9. No active classification model -> 404
10. No active anomaly model -> 404
11. Missing required feature -> 400
12. Target column supplied as feature -> rejected (400)
13. Invalid feature value -> rejected (400)
14. Actual prediction is generated from saved artifact
15. Returned model ID/version matches active registry model
16. Prediction history persistence
17. Prediction history listing is user-scoped
18. Cross-user prediction history access -> 404
19. Artifact missing -> controlled 500
20. Artifact path traversal -> rejected (400)
21. Empty batch -> 400
22. Batch size limit -> 400
23. Output order matches input order
24. Classification label decoding uses persisted encoder
25. Anomaly result correctly maps Isolation Forest -1/1
26. Model cache does not cross user/model boundaries
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
from app.db.models import ModelRegistryRecord, ModelPredictionRecord
from app.services.inference_service import clear_model_cache

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


def train_register_and_activate(
    client: TestClient,
    user: str,
    task: str = "rul_regression",
    model_type: str = "rf_regressor",
    name: str = "Model",
) -> dict:
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
        target = "rul"
        features = ["cycle", "vibration", "temperature", "pressure"]

    ds_id = upload_dataset(client, user, csv_data, f"DS_{uuid.uuid4().hex[:6]}")
    train_payload = {
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
    train_resp = client.post("/api/training/jobs", headers=auth_header(user), json=train_payload)
    assert train_resp.status_code == 201, train_resp.text
    job_id = train_resp.json()["id"]

    reg_resp = client.post(
        "/api/models/register",
        headers=auth_header(user),
        json={"training_job_id": job_id, "name": name},
    )
    assert reg_resp.status_code == 201, reg_resp.text
    model_id = reg_resp.json()["id"]

    act_resp = client.post(f"/api/models/{model_id}/activate", headers=auth_header(user))
    assert act_resp.status_code == 200, act_resp.text

    return {
        "model_id": model_id,
        "job_id": job_id,
        "features": features,
        "target": target,
        "task": task,
        "model_type": model_type,
    }


# ---------------------------------------------------------------------------
# 1. RUL inference with active RF model
# ---------------------------------------------------------------------------
def test_rul_inference_with_active_rf_model():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "rul_regression", "rf_regressor", "RUL RF")

        payload = {
            "features": {
                "cycle": 10,
                "vibration": 655.0,
                "temperature": 1600.0,
                "pressure": 1412.0,
            }
        }
        resp = client.post("/api/inference/rul", headers=auth_header(user), json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["task"] == "rul_regression"
        assert "prediction" in data
        assert "rul_cycles" in data["prediction"]
        assert isinstance(data["prediction"]["rul_cycles"], (int, float))
        assert data["model"]["id"] == info["model_id"]
        assert data["model"]["model_type"] == "rf_regressor"


# ---------------------------------------------------------------------------
# 2. RUL inference with active LSTM model
# ---------------------------------------------------------------------------
def test_rul_inference_with_active_lstm_model():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "rul_regression", "lstm_regressor", "RUL LSTM")

        payload = {
            "features": {
                "cycle": 15,
                "vibration": 662.0,
                "temperature": 1610.0,
                "pressure": 1418.0,
            }
        }
        resp = client.post("/api/inference/rul", headers=auth_header(user), json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["task"] == "rul_regression"
        assert "rul_cycles" in data["prediction"]
        assert isinstance(data["prediction"]["rul_cycles"], (int, float))
        assert data["model"]["model_type"] == "lstm_regressor"


# ---------------------------------------------------------------------------
# 3. Classification inference
# ---------------------------------------------------------------------------
def test_classification_inference():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "condition_classification", "rf_classifier", "Condition RF")

        payload = {
            "features": {
                "cycle": 5,
                "vibration": 620.0,
                "temperature": 1530.0,
                "pressure": 1315.0,
            }
        }
        resp = client.post("/api/inference/classification", headers=auth_header(user), json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["task"] == "condition_classification"
        assert "condition" in data["prediction"]
        assert data["prediction"]["condition"] in {"normal", "warning", "fault"}
        assert "probabilities" in data["prediction"]
        assert data["model"]["id"] == info["model_id"]


# ---------------------------------------------------------------------------
# 4. Anomaly inference
# ---------------------------------------------------------------------------
def test_anomaly_inference():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "anomaly_detection", "isolation_forest", "Anomaly ISO")

        payload = {
            "features": {
                "vibration": 642.0,
                "temperature": 1582.0,
                "pressure": 1402.0,
            }
        }
        resp = client.post("/api/inference/anomaly", headers=auth_header(user), json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["task"] == "anomaly_detection"
        assert "is_anomaly" in data["prediction"]
        assert isinstance(data["prediction"]["is_anomaly"], bool)
        assert "anomaly_score" in data["prediction"]
        assert isinstance(data["prediction"]["anomaly_score"], float)
        assert data["model"]["id"] == info["model_id"]


# ---------------------------------------------------------------------------
# 5. Batch RUL inference
# ---------------------------------------------------------------------------
def test_batch_rul_inference():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Batch RUL")

        rows = [
            {"cycle": 1, "vibration": 641.5, "temperature": 1582.0, "pressure": 1401.2},
            {"cycle": 20, "vibration": 670.0, "temperature": 1620.0, "pressure": 1424.0},
        ]
        resp = client.post("/api/inference/rul_regression/batch", headers=auth_header(user), json={"rows": rows})
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["task"] == "rul_regression"
        assert len(data["predictions"]) == 2
        assert "rul_cycles" in data["predictions"][0]
        assert "rul_cycles" in data["predictions"][1]
        assert data["total_rows"] == 2


# ---------------------------------------------------------------------------
# 6. Batch classification inference
# ---------------------------------------------------------------------------
def test_batch_classification_inference():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "condition_classification", "rf_classifier", "Batch CLS")

        rows = [
            {"cycle": 1, "vibration": 605.0, "temperature": 1510.0, "pressure": 1302.0},
            {"cycle": 2, "vibration": 625.0, "temperature": 1540.0, "pressure": 1320.0},
        ]
        resp = client.post("/api/inference/condition_classification/batch", headers=auth_header(user), json={"rows": rows})
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert len(data["predictions"]) == 2
        assert "condition" in data["predictions"][0]
        assert "condition" in data["predictions"][1]


# ---------------------------------------------------------------------------
# 7. Batch anomaly inference
# ---------------------------------------------------------------------------
def test_batch_anomaly_inference():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "anomaly_detection", "isolation_forest", "Batch Anomaly")

        rows = [
            {"vibration": 640.0, "temperature": 1580.0, "pressure": 1400.0},
            {"vibration": 999.0, "temperature": 2500.0, "pressure": 2000.0},
        ]
        resp = client.post("/api/inference/anomaly_detection/batch", headers=auth_header(user), json={"rows": rows})
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert len(data["predictions"]) == 2
        assert "is_anomaly" in data["predictions"][0]
        assert "is_anomaly" in data["predictions"][1]


# ---------------------------------------------------------------------------
# 8. No active RUL model -> 404
# ---------------------------------------------------------------------------
def test_no_active_rul_model_returns_404():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        resp = client.post("/api/inference/rul", headers=auth_header(user), json={"features": {"cycle": 1}})
        assert resp.status_code == 404
        assert "no active model" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 9. No active classification model -> 404
# ---------------------------------------------------------------------------
def test_no_active_classification_model_returns_404():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        resp = client.post("/api/inference/classification", headers=auth_header(user), json={"features": {"cycle": 1}})
        assert resp.status_code == 404
        assert "no active model" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 10. No active anomaly model -> 404
# ---------------------------------------------------------------------------
def test_no_active_anomaly_model_returns_404():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        resp = client.post("/api/inference/anomaly", headers=auth_header(user), json={"features": {"cycle": 1}})
        assert resp.status_code == 404
        assert "no active model" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 11. Missing required feature -> 400
# ---------------------------------------------------------------------------
def test_missing_required_feature_returns_400():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Model")

        # Missing 'vibration' and 'pressure'
        payload = {"features": {"cycle": 10, "temperature": 1600.0}}
        resp = client.post("/api/inference/rul", headers=auth_header(user), json=payload)
        assert resp.status_code == 400
        assert "missing required feature columns" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 12. Target column supplied as feature -> rejected (400)
# ---------------------------------------------------------------------------
def test_target_column_supplied_as_feature_rejected():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Model")

        payload = {
            "features": {
                "cycle": 10,
                "vibration": 655.0,
                "temperature": 1600.0,
                "pressure": 1412.0,
                "rul": 50,  # Target column supplied!
            }
        }
        resp = client.post("/api/inference/rul", headers=auth_header(user), json=payload)
        assert resp.status_code == 400
        assert "target column" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 13. Invalid feature value -> rejected (400)
# ---------------------------------------------------------------------------
def test_invalid_feature_value_rejected():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Model")

        payload = {
            "features": {
                "cycle": "not_a_number_abc",
                "vibration": 655.0,
                "temperature": 1600.0,
                "pressure": 1412.0,
            }
        }
        resp = client.post("/api/inference/rul", headers=auth_header(user), json=payload)
        assert resp.status_code == 400
        assert "invalid non-numeric value" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 14. Actual prediction is generated from saved artifact
# ---------------------------------------------------------------------------
def test_actual_prediction_generated_from_saved_artifact():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Math Test Model")

        p1 = {"features": {"cycle": 1, "vibration": 641.5, "temperature": 1582.0, "pressure": 1401.2}}
        p2 = {"features": {"cycle": 30, "vibration": 685.0, "temperature": 1640.0, "pressure": 1436.0}}

        r1 = client.post("/api/inference/rul", headers=auth_header(user), json=p1).json()
        r2 = client.post("/api/inference/rul", headers=auth_header(user), json=p2).json()

        # In our linear data generator, higher cycle/vibration gives smaller RUL
        # The trained model should reflect this relationship (r1 > r2)
        assert r1["prediction"]["rul_cycles"] != r2["prediction"]["rul_cycles"]
        assert r1["prediction"]["rul_cycles"] > r2["prediction"]["rul_cycles"]


# ---------------------------------------------------------------------------
# 15. Returned model ID/version matches active registry model
# ---------------------------------------------------------------------------
def test_returned_model_id_version_matches_active_registry_model():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Audit Ver Model")

        resp = client.post(
            "/api/inference/rul",
            headers=auth_header(user),
            json={"features": {"cycle": 1, "vibration": 640.0, "temperature": 1580.0, "pressure": 1400.0}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"]["id"] == info["model_id"]
        assert data["model"]["name"] == "Audit Ver Model"
        assert data["model"]["version"] == 1


# ---------------------------------------------------------------------------
# 16. Prediction history persistence
# ---------------------------------------------------------------------------
def test_prediction_history_persistence():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "rul_regression", "rf_regressor", "History Test")

        client.post(
            "/api/inference/rul",
            headers=auth_header(user),
            json={"features": {"cycle": 5, "vibration": 647.5, "temperature": 1590.0, "pressure": 1406.0}},
        )

        resp = client.get("/api/inference/history", headers=auth_header(user))
        assert resp.status_code == 200
        hist = resp.json()
        assert "predictions" in hist
        assert len(hist["predictions"]) >= 1
        p = hist["predictions"][0]
        assert p["task"] == "rul_regression"
        assert "prediction" in p
        assert "input_metadata" in p


# ---------------------------------------------------------------------------
# 17. Prediction history listing is user-scoped
# ---------------------------------------------------------------------------
def test_prediction_history_listing_is_user_scoped():
    with TestClient(app) as client:
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        train_register_and_activate(client, user_a, "rul_regression", "rf_regressor", "Model A")
        client.post(
            "/api/inference/rul",
            headers=auth_header(user_a),
            json={"features": {"cycle": 5, "vibration": 647.5, "temperature": 1590.0, "pressure": 1406.0}},
        )

        # User B has not run any predictions
        resp_b = client.get("/api/inference/history", headers=auth_header(user_b))
        assert resp_b.status_code == 200
        assert len(resp_b.json()["predictions"]) == 0


# ---------------------------------------------------------------------------
# 18. Cross-user prediction history access -> 404
# ---------------------------------------------------------------------------
def test_cross_user_prediction_history_access_returns_404():
    with TestClient(app) as client:
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        train_register_and_activate(client, user_a, "rul_regression", "rf_regressor", "Model A")
        client.post(
            "/api/inference/rul",
            headers=auth_header(user_a),
            json={"features": {"cycle": 5, "vibration": 647.5, "temperature": 1590.0, "pressure": 1406.0}},
        )

        hist_a = client.get("/api/inference/history", headers=auth_header(user_a)).json()
        pred_id = hist_a["predictions"][0]["id"]

        # User B attempts to access User A's prediction
        resp_b = client.get(f"/api/inference/history/{pred_id}", headers=auth_header(user_b))
        assert resp_b.status_code == 404


# ---------------------------------------------------------------------------
# 19. Artifact missing -> controlled 500
# ---------------------------------------------------------------------------
def test_artifact_missing_controlled_500():
    with TestClient(app) as client:
        clear_model_cache()
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Missing Artifact")

        # Delete artifact file
        db = next(get_db())
        try:
            model = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == info["model_id"]).first()
            if model and model.artifact_path and os.path.exists(model.artifact_path):
                os.remove(model.artifact_path)
        finally:
            db.close()

        resp = client.post(
            "/api/inference/rul",
            headers=auth_header(user),
            json={"features": {"cycle": 1, "vibration": 640.0, "temperature": 1580.0, "pressure": 1400.0}},
        )
        assert resp.status_code == 500
        assert "missing from storage" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 20. Artifact path traversal -> rejected (400)
# ---------------------------------------------------------------------------
def test_artifact_path_traversal_rejected():
    with TestClient(app) as client:
        clear_model_cache()
        user = f"user_{uuid.uuid4().hex[:8]}"
        info = train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Traversal Test")

        # Set malicious artifact path
        db = next(get_db())
        try:
            model = db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == info["model_id"]).first()
            if model:
                model.artifact_path = "../../etc/passwd"
                db.commit()
        finally:
            db.close()

        resp = client.post(
            "/api/inference/rul",
            headers=auth_header(user),
            json={"features": {"cycle": 1, "vibration": 640.0, "temperature": 1580.0, "pressure": 1400.0}},
        )
        assert resp.status_code == 400
        assert "traversal" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 21. Empty batch -> 400
# ---------------------------------------------------------------------------
def test_empty_batch_returns_400():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Batch Model")

        resp = client.post(
            "/api/inference/rul_regression/batch",
            headers=auth_header(user),
            json={"rows": []},
        )
        assert resp.status_code == 400
        assert "non-empty" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 22. Batch size limit -> 400
# ---------------------------------------------------------------------------
def test_batch_size_limit_returns_400():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Batch Limit Model")

        # Create batch of 501 rows exceeding default limit of 500
        rows = [
            {"cycle": 1, "vibration": 640.0, "temperature": 1580.0, "pressure": 1400.0}
            for _ in range(501)
        ]
        resp = client.post(
            "/api/inference/rul_regression/batch",
            headers=auth_header(user),
            json={"rows": rows},
        )
        assert resp.status_code == 400
        assert "exceeds maximum allowed" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 23. Output order matches input order
# ---------------------------------------------------------------------------
def test_output_order_matches_input_order():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "rul_regression", "rf_regressor", "Order Model")

        rows = [
            {"cycle": 1, "vibration": 641.0, "temperature": 1582.0, "pressure": 1401.0},
            {"cycle": 30, "vibration": 685.0, "temperature": 1640.0, "pressure": 1436.0},
            {"cycle": 5, "vibration": 647.0, "temperature": 1590.0, "pressure": 1406.0},
        ]
        resp = client.post("/api/inference/rul_regression/batch", headers=auth_header(user), json={"rows": rows})
        assert resp.status_code == 200
        preds = resp.json()["predictions"]

        # Predictions for cycle 1 > cycle 30
        assert preds[0]["rul_cycles"] > preds[1]["rul_cycles"]
        # Predictions for cycle 5 is between cycle 1 and cycle 30
        assert preds[0]["rul_cycles"] > preds[2]["rul_cycles"] > preds[1]["rul_cycles"]


# ---------------------------------------------------------------------------
# 24. Classification label decoding uses persisted encoder
# ---------------------------------------------------------------------------
def test_classification_label_decoding_uses_persisted_encoder():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "condition_classification", "rf_classifier", "Label Test Model")

        resp = client.post(
            "/api/inference/classification",
            headers=auth_header(user),
            json={"features": {"cycle": 1, "vibration": 600.0, "temperature": 1500.0, "pressure": 1300.0}},
        )
        assert resp.status_code == 200
        cond = resp.json()["prediction"]["condition"]
        # Decoded back into string label from original classes
        assert isinstance(cond, str)
        assert cond in {"normal", "warning", "fault"}


# ---------------------------------------------------------------------------
# 25. Anomaly result correctly maps Isolation Forest -1/1
# ---------------------------------------------------------------------------
def test_anomaly_result_correctly_maps_isolation_forest():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        train_register_and_activate(client, user, "anomaly_detection", "isolation_forest", "Iso Mapping Test")

        # Normal sample near training distribution center
        resp_norm = client.post(
            "/api/inference/anomaly",
            headers=auth_header(user),
            json={"features": {"vibration": 640.0, "temperature": 1580.0, "pressure": 1400.0}},
        )
        assert resp_norm.status_code == 200
        assert "is_anomaly" in resp_norm.json()["prediction"]

        # Extreme outlier sample far beyond training data
        resp_outlier = client.post(
            "/api/inference/anomaly",
            headers=auth_header(user),
            json={"features": {"vibration": 9999.0, "temperature": 9999.0, "pressure": 9999.0}},
        )
        assert resp_outlier.status_code == 200
        assert resp_outlier.json()["prediction"]["is_anomaly"] is True
        assert resp_outlier.json()["prediction"]["anomaly_score"] < 0


# ---------------------------------------------------------------------------
# 26. Model cache does not cross user/model boundaries
# ---------------------------------------------------------------------------
def test_model_cache_does_not_cross_user_model_boundaries():
    with TestClient(app) as client:
        clear_model_cache()
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        info_a = train_register_and_activate(client, user_a, "rul_regression", "rf_regressor", "Model User A")
        info_b = train_register_and_activate(client, user_b, "rul_regression", "rf_regressor", "Model User B")

        # Request A -> warms cache for User A
        r_a = client.post(
            "/api/inference/rul",
            headers=auth_header(user_a),
            json={"features": {"cycle": 1, "vibration": 640.0, "temperature": 1580.0, "pressure": 1400.0}},
        )
        assert r_a.json()["model"]["id"] == info_a["model_id"]

        # Request B -> must return User B's model, not User A's cached model
        r_b = client.post(
            "/api/inference/rul",
            headers=auth_header(user_b),
            json={"features": {"cycle": 1, "vibration": 640.0, "temperature": 1580.0, "pressure": 1400.0}},
        )
        assert r_b.json()["model"]["id"] == info_b["model_id"]
