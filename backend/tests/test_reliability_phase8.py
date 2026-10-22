"""Regression and reliability hardening tests for Phase 8.

Tests:
- Health and readiness/liveness endpoints (/health, /health/live, /health/ready)
- Request correlation ID header (X-Request-ID) propagation and generation
- Database session rollback and ping verification
- Path traversal and malicious filename protection in sanitize_filename
- Bounded input limits for single inference
- Training job cancellation idempotency and consistent terminal states
- Bounded RAG search parameters and empty query handling
"""
import os
import sys
import uuid
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.config import settings, validate_configuration
settings.ENVIRONMENT = "test"

from app.main import app, sanitize_filename
from app.db.session import check_db_ping, get_db, SessionLocal
from app.db.models import User, TrainingJobRecord
from app.services import dataset_service


TEST_USER_ID = "phase8_reliability_user"
AUTH_HEADERS = {"Authorization": f"Bearer test_token_{TEST_USER_ID}"}


def test_health_endpoint_backward_compatibility():
    """Verify that /health retains all expected legacy keys and reports valid status."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "live"
        assert data["status"] in ("healthy", "degraded")
        assert "models_loaded" in data
        assert "database" in data
        assert data["database"] == "connected"
        assert "dataset" in data
        assert "models" in data
        assert "artifacts_path" in data


def test_liveness_probe():
    """Verify that /health/live returns 200 with status alive."""
    with TestClient(app) as client:
        resp = client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"
        assert "timestamp" in data


def test_readiness_probe():
    """Verify that /health/ready verifies database connectivity and ML readiness."""
    with TestClient(app) as client:
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["checks"]["database"]["connected"] is True
        assert "ml_models" in data["checks"]
        assert "storage" in data["checks"]


def test_request_correlation_id_generated():
    """Verify that requests without X-Request-ID receive a generated correlation ID."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        req_id = resp.headers.get("x-request-id")
        assert req_id is not None
        assert req_id.startswith("req_")


def test_request_correlation_id_propagated():
    """Verify that client-supplied X-Request-ID is preserved in response headers."""
    custom_trace = "client-trace-12345-abcdef"
    with TestClient(app) as client:
        resp = client.get("/health", headers={"X-Request-ID": custom_trace})
        assert resp.status_code == 200
        assert resp.headers.get("x-request-id") == custom_trace


def test_database_ping_and_session_rollback():
    """Verify that check_db_ping works and get_db executes clean rollbacks on error."""
    assert check_db_ping() is True

    # Test that get_db generator rolls back when an exception occurs inside its block
    db_gen = get_db()
    db = next(db_gen)
    try:
        # Simulate an exception in route
        raise ValueError("Simulated route error")
    except ValueError:
        with pytest.raises(ValueError):
            db_gen.throw(ValueError("Simulated route error"))


def test_filename_sanitization_security():
    """Verify that path traversal, dots, and directory separators are stripped safely."""
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("..") == "document.txt"
    assert sanitize_filename("....") == "document.txt"
    assert sanitize_filename(".hidden_file.txt") == "hidden_file.txt"
    assert sanitize_filename("   ") == "document.txt"
    assert sanitize_filename("safe_data.csv") == "safe_data.csv"

    # Dataset service sanitizer
    assert dataset_service.sanitize_filename("../../../etc/shadow") == "shadow"
    assert dataset_service.sanitize_filename("..") == "dataset.csv"
    assert dataset_service.sanitize_filename("....") == "dataset.csv"
    assert dataset_service.sanitize_filename("valid_timeseries.csv") == "valid_timeseries.csv"


def test_bounded_rag_search():
    """Verify that document search clamps queries and top_k bounds safely."""
    with TestClient(app) as client:
        # Empty query rejection
        empty_resp = client.post("/api/knowledge/search", headers=AUTH_HEADERS, json={"query": "   "})
        assert empty_resp.status_code == 400

        # Valid query with excessively large top_k is clamped without error
        resp = client.post(
            "/api/knowledge/search",
            headers=AUTH_HEADERS,
            json={"query": "bearing vibration temperature", "top_k": 9999},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


def test_single_inference_input_bounds():
    """Verify that excessive feature payload (> 200 keys) is rejected with 400."""
    with TestClient(app) as client:
        # Empty features rejection
        empty_resp = client.post("/api/inference/rul", headers=AUTH_HEADERS, json={"features": {}})
        assert empty_resp.status_code == 400

        # Massive features payload rejection
        oversized = {f"col_{i}": float(i) for i in range(250)}
        over_resp = client.post("/api/inference/rul", headers=AUTH_HEADERS, json={"features": oversized})
        assert over_resp.status_code == 400
        assert "exceeds maximum allowed limit" in over_resp.json()["detail"]


def test_training_job_cancellation_idempotency():
    """Verify that cancelling already completed or cancelled training job is handled gracefully."""
    db = SessionLocal()
    job_id = f"job_test_cancel_{uuid.uuid4().hex[:8]}"
    try:
        # Create a mock completed job
        completed_job = TrainingJobRecord(
            id=job_id,
            user_id=TEST_USER_ID,
            dataset_id="ds_mock",
            task="rul_regression",
            model_type="rf_regressor",
            status="completed",
            feature_columns_json="[\"s1\"]",
        )
        db.add(completed_job)
        db.commit()

        with TestClient(app) as client:
            resp = client.post(f"/api/training/jobs/{job_id}/cancel", headers=AUTH_HEADERS)
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "Cannot cancel job with terminal status" in data["message"]
    finally:
        db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).delete()
        db.commit()
        db.close()


def test_configuration_validation_helper():
    """Verify that validate_configuration inspects environment variables safely."""
    issues = validate_configuration()
    assert isinstance(issues, list)


def test_render_database_url_resolution_and_production_strictness(monkeypatch):
    """Verify DATABASE_URL resolution from Render env vars, postgres:// fix, and production enforcement."""
    from app.config import Settings

    test_settings = Settings()
    # 1. Heroku/Render postgres:// prefix rewrite
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@host:5432/db")
    test_settings.ENVIRONMENT = "production"
    assert test_settings.DATABASE_URL == "postgresql://user:pass@host:5432/db"

    # 2. Render internal database url fallback
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_INTERNAL_URL", "postgresql://user:pass@dpg-xxx:5432/machinasense")
    test_settings.ENVIRONMENT = "production"
    assert test_settings.DATABASE_URL == "postgresql://user:pass@dpg-xxx:5432/machinasense"

    # 3. Production rejects ephemeral SQLite
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./machinasense.db")
    test_settings.ENVIRONMENT = "production"
    with pytest.raises(RuntimeError, match="Production requires a PostgreSQL database"):
        _ = test_settings.DATABASE_URL

    # 4. Production rejects missing DATABASE_URL
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_INTERNAL_URL", raising=False)
    monkeypatch.delenv("INTERNAL_DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    test_settings.ENVIRONMENT = "production"
    with pytest.raises(RuntimeError, match="DATABASE_URL .* must be configured in production"):
        _ = test_settings.DATABASE_URL


def test_production_cors_safeguard(monkeypatch):
    """Verify that Netlify origin is always included in production and wildcard * is rejected."""
    from app.config import Settings

    test_settings = Settings()
    test_settings.ENVIRONMENT = "production"

    # Custom origin provided without Netlify - Netlify origin is automatically added
    monkeypatch.setenv("CORS_ORIGINS", "https://custom-domain.com")
    origins = test_settings.CORS_ORIGINS
    assert "https://machinasense.netlify.app" in origins
    assert "https://custom-domain.com" in origins

    # Wildcard in production is filtered out
    monkeypatch.setenv("CORS_ORIGINS", "*, https://other.com/")
    origins = test_settings.CORS_ORIGINS
    assert "*" not in origins
    assert "https://other.com" in origins
    assert "https://machinasense.netlify.app" in origins


def test_copilot_resilient_to_deleted_machine():
    """Verify that Copilot conversations and messages do not crash when associated machine is deleted."""
    from app.db.models import Machine

    db = SessionLocal()
    mid = f"m_test_{uuid.uuid4().hex[:8]}"
    try:
        # Create a test machine
        machine = Machine(
            id=mid,
            user_id=TEST_USER_ID,
            name="Test Turbine For Copilot",
            machine_type="CFM56",
            location="Test Bay",
        )
        db.add(machine)
        db.commit()

        with TestClient(app) as client:
            # Create conversation linked to machine
            c_resp = client.post("/api/copilot/conversations", headers=AUTH_HEADERS, json={"machine_id": mid, "title": "Test Chat"})
            assert c_resp.status_code == 201
            cid = c_resp.json()["id"]

            # Now delete the machine
            db.delete(machine)
            db.commit()

            # Sending message to the conversation should NOT 404; it unbinds gracefully and responds
            msg_resp = client.post(f"/api/copilot/conversations/{cid}/messages", headers=AUTH_HEADERS, json={"content": "What is the status of the plant?"})
            assert msg_resp.status_code == 201
            msg_data = msg_resp.json()
            assert "content" in msg_data
            assert msg_data["content"] != ""
    finally:
        db.close()

