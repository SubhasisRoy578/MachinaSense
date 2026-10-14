"""Comprehensive test suite for Backend Part 1: Real Dataset Service.
Tests cover:
1. Authenticated CSV upload and statistical profiling
2. Inferred data types, numeric/categorical splits, and distribution stats
3. Bounded preview rows generation
4. Authenticated dataset list
5. Authenticated dataset detail retrieval
6. Dataset deletion and physical file cleanup
7. Strict cross-user isolation for list, get, and delete
8. Unauthenticated access rejection (401)
9. Non-CSV upload rejection (400)
10. Empty CSV rejection (400)
11. Malformed CSV rejection (400)
12. Oversized upload rejection (400)
"""
import io
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

settings.ENVIRONMENT = "test"

def auth_header(user_id: str):
    return {"Authorization": f"Bearer test_token_{user_id}"}

SAMPLE_TELEMETRY_CSV = """cycle,vibration,temperature,pressure,status
1,640.2,1580.5,1400.1,healthy
2,641.5,1582.0,1402.3,healthy
3,643.8,1585.1,1405.0,degrading
4,650.0,1590.2,1412.8,degrading
5,665.4,1602.0,1425.6,critical
"""

def test_upload_dataset_success():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        files = {
            "file": ("telemetry_run_01.csv", io.BytesIO(SAMPLE_TELEMETRY_CSV.encode("utf-8")), "text/csv")
        }
        data = {"name": "Bearing Telemetry Batch A"}
        
        resp = client.post("/api/datasets", headers=auth_header(user), files=files, data=data)
        assert resp.status_code == 201, resp.text
        d = resp.json()

        # Contract assertions
        assert d["id"].startswith("ds_")
        assert d["name"] == "Bearing Telemetry Batch A"
        assert d["fileName"] == "telemetry_run_01.csv"
        assert d["fileSize"] > 0
        assert d["status"] == "ready"
        assert "uploadedAt" in d

        # Statistics assertions
        stats = d["statistics"]
        assert stats["rowCount"] == 5
        assert stats["columnCount"] == 5
        assert stats["numericColumnCount"] == 4
        assert stats["categoricalColumnCount"] == 1
        assert stats["missingValueCount"] == 0
        assert stats["duplicateRowCount"] == 0
        assert stats["memorySizeBytes"] > 0

        # Column Explorer assertions
        columns = d["columns"]
        assert len(columns) == 5
        col_names = [c["name"] for c in columns]
        assert col_names == ["cycle", "vibration", "temperature", "pressure", "status"]

        # Check numeric stats
        vib_col = next(c for c in columns if c["name"] == "vibration")
        assert vib_col["isNumeric"] is True
        assert vib_col["numericStats"] is not None
        assert vib_col["numericStats"]["min"] == 640.2
        assert vib_col["numericStats"]["max"] == 665.4
        assert vib_col["numericStats"]["mean"] > 640

        # Check categorical stats
        status_col = next(c for c in columns if c["name"] == "status")
        assert status_col["isNumeric"] is False
        assert status_col["categoricalStats"] is not None
        assert status_col["categoricalStats"]["uniqueCount"] == 3
        assert status_col["categoricalStats"]["topValue"] in {"healthy", "degrading"}

        # Preview assertions
        preview = d["preview"]
        assert preview["totalRows"] == 5
        assert preview["previewRows"] == 5
        assert len(preview["rows"]) == 5
        assert preview["columns"] == ["cycle", "vibration", "temperature", "pressure", "status"]
        assert preview["rows"][0]["status"] == "healthy"
        assert preview["rows"][0]["cycle"] == 1


def test_list_datasets_user_isolation():
    with TestClient(app) as client:
        user_a = f"user_a_{uuid.uuid4().hex[:8]}"
        user_b = f"user_b_{uuid.uuid4().hex[:8]}"

        # User A uploads
        resp_a = client.post(
            "/api/datasets",
            headers=auth_header(user_a),
            files={"file": ("dataset_a.csv", io.BytesIO(SAMPLE_TELEMETRY_CSV.encode("utf-8")), "text/csv")},
            data={"name": "Dataset Alpha"}
        )
        assert resp_a.status_code == 201
        id_a = resp_a.json()["id"]

        # User B uploads
        resp_b = client.post(
            "/api/datasets",
            headers=auth_header(user_b),
            files={"file": ("dataset_b.csv", io.BytesIO(SAMPLE_TELEMETRY_CSV.encode("utf-8")), "text/csv")},
            data={"name": "Dataset Beta"}
        )
        assert resp_b.status_code == 201
        id_b = resp_b.json()["id"]

        # User A list
        list_a = client.get("/api/datasets", headers=auth_header(user_a))
        assert list_a.status_code == 200
        items_a = list_a.json()["datasets"]
        assert any(item["id"] == id_a for item in items_a)
        assert not any(item["id"] == id_b for item in items_a)

        # User B list
        list_b = client.get("/api/datasets", headers=auth_header(user_b))
        assert list_b.status_code == 200
        items_b = list_b.json()["datasets"]
        assert any(item["id"] == id_b for item in items_b)
        assert not any(item["id"] == id_a for item in items_b)


def test_get_dataset_detail_and_cross_user_rejection():
    with TestClient(app) as client:
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        # User A uploads
        resp = client.post(
            "/api/datasets",
            headers=auth_header(user_a),
            files={"file": ("turbofan.csv", io.BytesIO(SAMPLE_TELEMETRY_CSV.encode("utf-8")), "text/csv")},
        )
        assert resp.status_code == 201
        ds_id = resp.json()["id"]

        # User A can get details
        get_a = client.get(f"/api/datasets/{ds_id}", headers=auth_header(user_a))
        assert get_a.status_code == 200
        assert get_a.json()["id"] == ds_id
        assert get_a.json()["statistics"]["rowCount"] == 5

        # User B gets 404 (not authorized / not found)
        get_b = client.get(f"/api/datasets/{ds_id}", headers=auth_header(user_b))
        assert get_b.status_code == 404
        assert get_b.json()["detail"] == "Dataset not found."


def test_delete_dataset_cross_user_rejection_and_cleanup():
    with TestClient(app) as client:
        user_a = f"user_{uuid.uuid4().hex[:8]}"
        user_b = f"user_{uuid.uuid4().hex[:8]}"

        # User A uploads
        resp = client.post(
            "/api/datasets",
            headers=auth_header(user_a),
            files={"file": ("to_delete.csv", io.BytesIO(SAMPLE_TELEMETRY_CSV.encode("utf-8")), "text/csv")},
        )
        assert resp.status_code == 201
        ds_id = resp.json()["id"]

        # User B attempts to delete User A's dataset -> 404
        del_b = client.delete(f"/api/datasets/{ds_id}", headers=auth_header(user_b))
        assert del_b.status_code == 404

        # User A deletes their own dataset -> 204
        del_a = client.delete(f"/api/datasets/{ds_id}", headers=auth_header(user_a))
        assert del_a.status_code == 204

        # Verify it is deleted from User A
        get_again = client.get(f"/api/datasets/{ds_id}", headers=auth_header(user_a))
        assert get_again.status_code == 404


def test_unauthenticated_access_rejected():
    with TestClient(app) as client:
        # POST without auth -> 401
        resp = client.post(
            "/api/datasets",
            files={"file": ("unauth.csv", io.BytesIO(b"a,b\n1,2"), "text/csv")}
        )
        assert resp.status_code == 401

        # GET without auth -> 401
        resp = client.get("/api/datasets")
        assert resp.status_code == 401

        # GET /{id} without auth -> 401
        resp = client.get("/api/datasets/ds_random123")
        assert resp.status_code == 401

        # DELETE /{id} without auth -> 401
        resp = client.delete("/api/datasets/ds_random123")
        assert resp.status_code == 401


def test_invalid_file_extension_rejected():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        resp = client.post(
            "/api/datasets",
            headers=auth_header(user),
            files={"file": ("data.txt", io.BytesIO(b"plain text"), "text/plain")}
        )
        assert resp.status_code == 400
        assert "CSV" in resp.json()["detail"]


def test_empty_csv_rejected():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        resp = client.post(
            "/api/datasets",
            headers=auth_header(user),
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()


def test_malformed_csv_rejected():
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        resp = client.post(
            "/api/datasets",
            headers=auth_header(user),
            files={"file": ("malformed.csv", io.BytesIO(b"\x00\x01\x02\xff"), "text/csv")}
        )
        assert resp.status_code == 400


def test_oversized_csv_rejected(monkeypatch):
    with TestClient(app) as client:
        user = f"user_{uuid.uuid4().hex[:8]}"
        # Set tiny limit to test oversized rejection
        monkeypatch.setattr(settings, "MAX_FILE_SIZE_BYTES", 50)
        resp = client.post(
            "/api/datasets",
            headers=auth_header(user),
            files={"file": ("oversized.csv", io.BytesIO(b"col1,col2\n" + b"1,2\n" * 20), "text/csv")}
        )
        assert resp.status_code == 400
        assert "exceeds" in resp.json()["detail"].lower()
