"""Phase 7 Backend Tests: Evidence-Backed Maintenance Decision Support Service."""
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
from app.db.models import (
    DocumentChunk,
    KnowledgeDoc,
    Machine,
    ModelPredictionRecord,
    Prediction,
    SensorReading,
    User,
)

settings.ENVIRONMENT = "test"


def auth_header(user_id: str):
    return {"Authorization": f"Bearer test_token_{user_id}"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_unauthenticated_decision_support_rejected(client):
    res = client.post("/api/decision-support", json={})
    assert res.status_code == 401


def test_decision_support_empty_payload(client):
    res = client.post("/api/decision-support", headers=auth_header("user_ds_1"), json={})
    assert res.status_code == 200
    data = res.json()
    assert "observed_evidence" in data
    assert "suggested_checks" in data
    assert "suggested_action" in data
    assert "uncertainties_and_limitations" in data
    assert "compliance_notice" in data
    # Advisory disclaimer check
    assert "advisory" in data["compliance_notice"].lower()


def test_decision_support_with_machine_and_rag(client):
    uid = f"ds_user_{uuid.uuid4().hex[:6]}"
    mid = f"mach_{uuid.uuid4().hex[:6]}"

    db = next(get_db())
    try:
        # Create user
        u = User(id=uid, email=f"{uid}@test.com", first_name="DS", last_name="User")
        db.add(u)

        # Create machine
        m = Machine(
            id=mid,
            user_id=uid,
            name="Turbofan Engine #7",
            machine_type="CFM-56 Turbofan",
            location="Test Bay 4",
        )
        db.add(m)

        # Add low RUL prediction (critical)
        p = Prediction(
            id=str(uuid.uuid4()),
            machine_id=mid,
            user_id=uid,
            predicted_rul=24.5,
            rf_predicted_rul=26.0,
            model_used="PyTorch LSTM Temporal RUL",
            confidence_min=18.0,
            confidence_max=31.0,
            health_score=25,
            risk_level="critical",
        )
        db.add(p)

        # Add recent sensor reading
        s = SensorReading(
            machine_id=mid,
            user_id=uid,
            s2=643.2,
            s3=1592.4,
            s4=1410.8,
            s7=553.9,
            s11=47.5,
            s14=8130.0,
            s17=392.0,
            cycle=185,
        )
        db.add(s)

        # Add technical document chunk to RAG
        doc_id = str(uuid.uuid4())
        doc = KnowledgeDoc(
            id=doc_id,
            user_id=uid,
            scope="USER_KNOWLEDGE",
            title="CFM56 Maintenance Manual Section 72",
            type="PDF",
            size=1024,
            source_category="Manual",
        )
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            user_id=uid,
            page=42,
            section="72-31-00 Compressor Inspection",
            content="When high pressure compressor outlet temperature (s3) exceeds threshold with rapid degradation cycles below 30, video borescope inspection of stage 4 stator vanes is mandatory.",
        )
        db.add_all([doc, chunk])
        db.commit()

        # Call decision support
        res = client.post(
            "/api/decision-support",
            headers=auth_header(uid),
            json={"machine_id": mid, "symptoms": "high compressor temperature and vibration"},
        )
        assert res.status_code == 200
        data = res.json()

        # 1. Observed evidence
        obs = data["observed_evidence"]
        assert obs["machine_id"] == mid
        assert obs["predicted_rul_cycles"] == 24.5
        assert obs["risk_level"] == "critical"
        assert "telemetry_observations" in obs
        assert obs["telemetry_observations"]["cycle"] == 185

        # 2. Supporting documentation from RAG
        docs = data["supporting_documentation"]
        assert len(docs) > 0
        citation = docs[0]
        assert citation["document_title"] == "CFM56 Maintenance Manual Section 72"
        assert citation["page"] == 42
        assert "72-31-00" in citation["section"]
        assert citation["relevance_score"] > 0

        # 3. Suggested checks
        checks = data["suggested_checks"]
        assert len(checks) > 0
        assert any("borescope" in c["procedure"].lower() for c in checks)

        # 4. Advisory action
        action = data["suggested_action"]
        assert action["action_type"] == "Immediate Scheduled Overhaul"
        assert action["requires_human_approval"] is True

        # 5. Non-existent machine returns 404
        res_bad = client.post(
            "/api/decision-support",
            headers=auth_header(uid),
            json={"machine_id": "non_existent_mach_999"},
        )
        assert res_bad.status_code == 404

        # 6. Another user cannot query this user's machine
        res_other = client.post(
            "/api/decision-support",
            headers=auth_header("other_unauthorized_user"),
            json={"machine_id": mid},
        )
        assert res_other.status_code == 404

    finally:
        db.close()
