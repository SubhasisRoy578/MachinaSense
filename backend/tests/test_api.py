import pytest
from fastapi.testclient import TestClient
import sys
import os

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        assert data["backend"] == "live"

def test_machines_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/machines")
        assert response.status_code == 200
        machines = response.json()
        assert isinstance(machines, list)
        assert len(machines) > 0
        m1 = machines[0]
        assert "id" in m1
        assert "healthScore" in m1
        assert "predictedRul" in m1

def test_machine_detail_and_404():
    with TestClient(app) as client:
        response = client.get("/api/machines/FD001-001")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "FD001-001"
        assert "sensors" in data

        # Invalid machine ID
        invalid_res = client.get("/api/machines/FD001-99999")
        assert invalid_res.status_code == 404

def test_machine_prediction():
    with TestClient(app) as client:
        response = client.get("/api/machines/FD001-001/prediction")
        assert response.status_code == 200
        data = response.json()
        assert "predictedRul" in data
        assert "confidenceInterval" in data
        assert "degradationCurve" in data

def test_machine_anomalies():
    with TestClient(app) as client:
        response = client.get("/api/machines/FD001-001/anomalies")
        assert response.status_code == 200
        data = response.json()
        assert "anomalyEvents" in data

def test_analytics_models():
    with TestClient(app) as client:
        response = client.get("/api/analytics/models")
        assert response.status_code == 200
        data = response.json()
        assert "evaluation" in data
        assert "rul_models" in data["evaluation"]

def test_predict_rul():
    with TestClient(app) as client:
        sample_features = [642.0, 1585.0, 1405.0, 21.6, 553.0, 2388.0, 9050.0, 47.5, 521.0, 2388.0, 8120.0, 8.4, 392.0, 38.8, 23.2]
        payload = {"features": sample_features}
        response = client.post("/api/predict/rul", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_rul" in data
        assert "confidence_interval" in data

def test_detect_anomaly():
    with TestClient(app) as client:
        sample_features = [642.0, 1585.0, 1405.0, 21.6, 553.0, 2388.0, 9050.0, 47.5, 521.0, 2388.0, 8120.0, 8.4, 392.0, 38.8, 23.2]
        payload = {"features": sample_features}
        response = client.post("/api/detect/anomaly", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "anomaly_score" in data
        assert "severity" in data

if __name__ == "__main__":
    pytest.main(["-v", __file__])
