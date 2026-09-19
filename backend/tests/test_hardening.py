import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app, sanitize_filename
from app.config import settings

client = TestClient(app)

# ---------------------------------------------------------
# 1. Configuration & Security Tests
# ---------------------------------------------------------
def test_config_defaults():
    assert settings.PROJECT_NAME == "MachinaSense Industrial Intelligence & Grounded RAG API"
    assert settings.PORT > 0
    assert len(settings.ALLOWED_EXTENSIONS) == 3
    assert ".pdf" in settings.ALLOWED_EXTENSIONS
    assert settings.MAX_FILE_SIZE_BYTES > 0

def test_filename_sanitization():
    # Prevent path traversal
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    # Clean non-alphanumeric characters except safe symbols
    assert sanitize_filename("my manual @#$% v1.pdf") == "my_manual______v1.pdf"
    assert sanitize_filename("") == "document.txt"

# ---------------------------------------------------------
# 2. Health & Machine APIs
# ---------------------------------------------------------
def test_health_detailed():
    with TestClient(app) as c:
        res = c.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["healthy", "degraded", "error"]
        assert "models_loaded" in data
        assert "rag_status" in data
        assert "llm_status" in data
        assert "models" in data
        assert len(data["models"]) == 4

def test_machine_invalid_404():
    with TestClient(app) as c:
        res = c.get("/api/machines/NON-EXISTENT-MACHINE")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

def test_machine_sensors_invalid_404():
    with TestClient(app) as c:
        res = c.get("/api/machines/INVALID-MACHINE-ID/sensors")
        assert res.status_code == 404

def test_machine_prediction_invalid_404():
    with TestClient(app) as c:
        res = c.get("/api/machines/INVALID-MACHINE-ID/prediction")
        assert res.status_code == 404

def test_machine_anomalies_invalid_404():
    with TestClient(app) as c:
        res = c.get("/api/machines/INVALID-MACHINE-ID/anomalies")
        assert res.status_code == 404

# ---------------------------------------------------------
# 3. RUL Prediction & Anomaly Hardening (Input Validation)
# ---------------------------------------------------------
def test_predict_rul_invalid_feature_length():
    with TestClient(app) as c:
        # Expected 15 features, pass 5
        res = c.post("/api/predict/rul", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert res.status_code == 400
        assert "Expected 15 features" in res.json()["detail"]

def test_predict_rul_empty_sequence():
    with TestClient(app) as c:
        res = c.post("/api/predict/rul", json={"sequence": []})
        assert res.status_code == 400
        assert "Sequence matrix cannot be empty" in res.json()["detail"]

def test_predict_rul_missing_body():
    with TestClient(app) as c:
        res = c.post("/api/predict/rul", json={})
        assert res.status_code == 400
        assert "Must provide 'features' or 'sequence'" in res.json()["detail"]

def test_detect_anomaly_invalid_feature_length():
    with TestClient(app) as c:
        # Pass 3 features instead of 15
        res = c.post("/api/detect/anomaly", json={"features": [10.0, 20.0, 30.0]})
        assert res.status_code == 400
        assert "Expected 15 features" in res.json()["detail"]

# ---------------------------------------------------------
# 4. Knowledge Base Hardening (Security & Validation)
# ---------------------------------------------------------
def test_upload_invalid_extension():
    with TestClient(app) as c:
        res = c.post(
            "/api/knowledge/upload",
            files={"file": ("malicious.exe", b"MZ\x90\x00BinaryPayload", "application/octet-stream")},
            data={"category": "Manual"}
        )
        assert res.status_code == 400
        assert "Unsupported file extension" in res.json()["detail"]

def test_upload_empty_file():
    with TestClient(app) as c:
        res = c.post(
            "/api/knowledge/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
            data={"category": "Manual"}
        )
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()

def test_search_empty_query():
    with TestClient(app) as c:
        res = c.post("/api/knowledge/search", json={"query": "   "})
        assert res.status_code == 400
        assert "cannot be empty" in res.json()["detail"]

def test_document_chunks_invalid_404():
    with TestClient(app) as c:
        res = c.get("/api/knowledge/documents/NON-EXISTENT-DOC/chunks")
        assert res.status_code == 404

# ---------------------------------------------------------
# 5. Diagnostics Hardening
# ---------------------------------------------------------
def test_create_diagnostic_invalid_machine():
    with TestClient(app) as c:
        res = c.post("/api/diagnostics", json={"machine_id": "UNKNOWN_MACHINE_999"})
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

def test_get_diagnostic_case_invalid_404():
    with TestClient(app) as c:
        res = c.get("/api/diagnostics/NON-EXISTENT-CASE-ID")
        assert res.status_code == 404
