"""Authentication, ownership, persistence, and ingestion regression tests."""
import os, sys, uuid
from fastapi.testclient import TestClient
BASE_DIR=os.path.normpath(os.path.join(os.path.dirname(__file__),".."));sys.path.insert(0,BASE_DIR)
from app.main import app
from app.config import settings
settings.ENVIRONMENT="test"

def headers(user): return {"Authorization":f"Bearer test_token_{user}"}
def create_machine(client,user):
    mid=f"test-{uuid.uuid4().hex[:10]}"
    response=client.post("/api/machines",headers=headers(user),json={"machine_id":mid,"name":"Test Pump","machine_type":"Pump","location":"Lab"})
    assert response.status_code==201
    return mid

def test_health_is_public():
    with TestClient(app) as client: assert client.get("/health").status_code==200
def test_protected_route_requires_authentication():
    with TestClient(app) as client: assert client.get("/api/machines").status_code==401
def test_machine_isolation_and_cross_user_denial():
    with TestClient(app) as client:
        mid=create_machine(client,"owner")
        assert client.get(f"/api/machines/{mid}",headers=headers("owner")).status_code==200
        assert client.get(f"/api/machines/{mid}",headers=headers("other")).status_code==404
def test_invalid_telemetry_is_rejected():
    with TestClient(app) as client:
        mid=create_machine(client,"csv")
        response=client.post(f"/api/machines/{mid}/telemetry",headers=headers("csv"),files={"file":("bad.csv",b"cycle,s2\n1,2\n","text/csv")})
        assert response.status_code==400
        assert "requires columns" in response.json()["detail"]
def test_copilot_conversation_is_isolated():
    with TestClient(app) as client:
        convo=client.post("/api/copilot/conversations",headers=headers("alice"),json={}).json()
        assert client.get(f"/api/copilot/conversations/{convo['id']}",headers=headers("bob")).status_code==404
def test_diagnostic_and_maintenance_ownership():
    with TestClient(app) as client:
        mid=create_machine(client,"workflow")
        diagnostic=client.post("/api/diagnostics",headers=headers("workflow"),json={"machine_id":mid})
        assert diagnostic.status_code==201
        assert client.get(f"/api/diagnostics/{diagnostic.json()['id']}",headers=headers("intruder")).status_code==404
        task=client.post("/api/maintenance",headers=headers("workflow"),json={"machine_id":mid,"title":"Inspect","description":"User-created action"})
        assert task.status_code==201
        assert client.delete(f"/api/maintenance/{task.json()['id']}",headers=headers("intruder")).status_code==404
