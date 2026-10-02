"""Comprehensive test suite for MachinaSense Part 3:
Production Data Integrity, Real Fleet Analytics, Database & Migrations,
CORS & Security Hardening, Error Handling, and Cross-User Isolation.
"""
import os, sys, uuid, json
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


def create_test_machine(client: TestClient, user_id: str) -> str:
    mid = f"m-{uuid.uuid4().hex[:8]}"
    resp = client.post(
        "/api/machines",
        headers=auth_header(user_id),
        json={"machine_id": mid, "name": f"Turbofan {mid}", "machine_type": "Turbofan Engine", "location": "Plant 1"},
    )
    assert resp.status_code == 201
    return mid


# 1. New/empty account fleet analytics returns 0 and null metrics (no mock/fake data)
def test_fleet_analytics_empty_account():
    with TestClient(app) as client:
        resp = client.get("/api/analytics/fleet", headers=auth_header("new_user_empty"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_machines"] == 0
        assert data["healthy"] == 0
        assert data["warning"] == 0
        assert data["critical"] == 0
        assert data["offline"] == 0
        assert data["average_rul"] is None
        assert data["total_anomalies"] == 0
        assert data["pending_maintenance"] == 0
        assert data["fleet"] == []


# 2. Populated fleet analytics calculates real metrics from user data
def test_fleet_analytics_populated():
    with TestClient(app) as client:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        m1 = create_test_machine(client, user_id)
        m2 = create_test_machine(client, user_id)

        # Create a maintenance task for m1
        client.post(
            "/api/maintenance",
            headers=auth_header(user_id),
            json={"machine_id": m1, "title": "Inspect Bearings", "description": "High vibration check", "priority": "high"},
        )

        resp = client.get("/api/analytics/fleet", headers=auth_header(user_id))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_machines"] == 2
        assert data["offline"] == 2  # no telemetry uploaded yet
        assert data["pending_maintenance"] == 1
        assert len(data["fleet"]) == 2
        machine_ids = [m["id"] for m in data["fleet"]]
        assert m1 in machine_ids
        assert m2 in machine_ids


# 3. Fleet analytics strict user tenant isolation
def test_fleet_analytics_user_isolation():
    with TestClient(app) as client:
        user_a = f"user_a_{uuid.uuid4().hex[:8]}"
        user_b = f"user_b_{uuid.uuid4().hex[:8]}"

        m_a = create_test_machine(client, user_a)

        # User A sees 1 machine
        resp_a = client.get("/api/analytics/fleet", headers=auth_header(user_a))
        assert resp_a.status_code == 200
        assert resp_a.json()["total_machines"] == 1

        # User B sees 0 machines
        resp_b = client.get("/api/analytics/fleet", headers=auth_header(user_b))
        assert resp_b.status_code == 200
        assert resp_b.json()["total_machines"] == 0
        assert resp_b.json()["fleet"] == []


# 4. Production database configuration disallows ephemeral SQLite
def test_production_sqlite_disallowed():
    orig_env = settings.ENVIRONMENT
    orig_url = settings._db_url
    try:
        settings.ENVIRONMENT = "production"
        settings._db_url = ""
        # Accessing DATABASE_URL without setting it in production must raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            _ = settings.DATABASE_URL
        assert "SQLite fallback is not permitted in production" in str(exc_info.value)
    finally:
        settings.ENVIRONMENT = orig_env
        settings._db_url = orig_url


# 5. Production CORS configuration rejects wildcard '*' and normalizes slashes
def test_production_cors_security():
    orig_env = settings.ENVIRONMENT
    orig_cors = settings._cors_env
    try:
        settings.ENVIRONMENT = "production"
        # Wildcard provided in production must be rejected and fallback to safe production origin
        settings._cors_env = "*"
        origins = settings.CORS_ORIGINS
        assert "*" not in origins
        assert "https://machinasense.netlify.app" in origins

        # Trailing slash is stripped
        settings._cors_env = "https://app.machinasense.com/,https://preview.machinasense.com"
        origins = settings.CORS_ORIGINS
        assert "https://app.machinasense.com" in origins
        assert "https://preview.machinasense.com" in origins
    finally:
        settings.ENVIRONMENT = orig_env
        settings._cors_env = orig_cors


# 6. Cross-user isolation on telemetry, sensor data, and prediction curves
def test_cross_user_telemetry_isolation():
    with TestClient(app) as client:
        owner = f"owner_{uuid.uuid4().hex[:8]}"
        intruder = f"intruder_{uuid.uuid4().hex[:8]}"
        mid = create_test_machine(client, owner)

        # Intruder attempts to fetch sensors or prediction curve for owner's machine
        resp_sensors = client.get(f"/api/machines/{mid}/sensors", headers=auth_header(intruder))
        assert resp_sensors.status_code == 404

        resp_pred = client.get(f"/api/machines/{mid}/prediction", headers=auth_header(intruder))
        assert resp_pred.status_code == 404

        resp_anom = client.get(f"/api/machines/{mid}/anomalies", headers=auth_header(intruder))
        assert resp_anom.status_code == 404


# 7. Cross-user isolation on Copilot message injection
def test_cross_user_copilot_message_injection():
    with TestClient(app) as client:
        alice = f"alice_{uuid.uuid4().hex[:8]}"
        bob = f"bob_{uuid.uuid4().hex[:8]}"

        # Alice creates a conversation
        convo = client.post("/api/copilot/conversations", headers=auth_header(alice), json={"title": "Private Convo"}).json()
        cid = convo["id"]

        # Bob attempts to send message to Alice's conversation
        resp = client.post(
            f"/api/copilot/conversations/{cid}/messages",
            headers=auth_header(bob),
            json={"content": "Malicious probe"},
        )
        assert resp.status_code == 404


# 8. Models analytics returns benchmark metadata and evaluation metrics
def test_models_analytics_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/analytics/models", headers=auth_header("test_user"))
        assert resp.status_code == 200
        data = resp.json()
        assert "NASA C-MAPSS FD001" in data["dataset"]
        assert "evaluation" in data
        assert "metadata" in data
        assert "disclaimer" in data


# 9. Health endpoint is public and reports accurate system statuses
def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "live"
        assert "models_loaded" in data
        assert data["rag_status"] == "ready"
        assert data["llm_status"] == "provider-chain"
