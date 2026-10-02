"""Comprehensive test suite for MachinaSense Part 2:
Copilot, Gemini/Grok provider chain, RAG, Diagnostics, and Maintenance.
"""
import os, sys, uuid, json
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app
from app.config import settings
from app.services.llm_service import LLMService, ProviderError

settings.ENVIRONMENT = "test"


def auth_header(user_id: str):
    return {"Authorization": f"Bearer test_token_{user_id}"}


class MockProvider:
    def __init__(self, name: str, result=None, fails: bool = False):
        self.name = name
        self.result = result
        self.fails = fails
        self.called = False

    def generate(self, context, question):
        self.called = True
        if self.fails:
            raise ProviderError(f"{self.name} provider simulation error")
        return self.result or f"{self.name} simulated output"


# Helper to create a machine for testing
def create_test_machine(client: TestClient, user_id: str) -> str:
    mid = f"m-{uuid.uuid4().hex[:8]}"
    resp = client.post(
        "/api/machines",
        headers=auth_header(user_id),
        json={"machine_id": mid, "name": f"Turbofan {mid}", "machine_type": "Turbofan Engine", "location": "Test Stand 1"},
    )
    assert resp.status_code == 201
    return mid


# 1. Copilot "hello"
def test_copilot_hello():
    llm = LLMService(MockProvider("gemini", fails=True), MockProvider("grok", fails=True))
    res = llm.generate_grounded_response({"machine": None, "ml_findings": "", "evidence": []}, "hello")
    assert "Insufficient machine data" not in res["content"]
    assert "MachinaSense Engineering Copilot" in res["content"]
    assert res["provider_used"] == "grounded_fallback"
    assert res["fallback_level"] == 2


# 2. Copilot general engineering question
def test_copilot_general_questions():
    llm = LLMService(MockProvider("gemini", fails=True), MockProvider("grok", fails=True))
    for q in ["what can you do?", "what is predictive maintenance?", "explain RUL", "explain anomaly detection", "explain remaining useful life"]:
        res = llm.generate_grounded_response({"machine": None, "ml_findings": "", "evidence": []}, q)
        assert "Insufficient machine data" not in res["content"]
        assert len(res["content"]) > 50
        assert res["provider_used"] == "grounded_fallback"
        assert res["fallback_level"] == 2


# 3. Copilot without machine
def test_copilot_without_machine_api():
    with TestClient(app) as client:
        # Create general conversation without a machine
        convo = client.post("/api/copilot/conversations", headers=auth_header("user_gen"), json={"title": "General Inquiries"}).json()
        assert convo["machineId"] is None
        
        # Send general message
        resp = client.post(
            f"/api/copilot/conversations/{convo['id']}/messages",
            headers=auth_header("user_gen"),
            json={"content": "what is predictive maintenance?"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "Predictive Maintenance" in data["content"] or "predictive maintenance" in data["content"].lower()
        assert "Insufficient machine data" not in data["content"]


# 4. Copilot with selected machine
def test_copilot_with_selected_machine():
    with TestClient(app) as client:
        mid = create_test_machine(client, "user_mach")
        # Create conversation with machine
        convo = client.post(
            "/api/copilot/conversations",
            headers=auth_header("user_mach"),
            json={"machine_id": mid, "title": f"Investigation for {mid}"},
        ).json()
        assert convo["machineId"] == mid

        # Changing machine on existing conversation
        mid2 = create_test_machine(client, "user_mach")
        patch_resp = client.patch(
            f"/api/copilot/conversations/{convo['id']}",
            headers=auth_header("user_mach"),
            json={"machine_id": mid2},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["machineId"] == mid2

        # Send message with selected machine
        msg_resp = client.post(
            f"/api/copilot/conversations/{convo['id']}/messages",
            headers=auth_header("user_mach"),
            json={"content": "what is the machine status?"},
        )
        assert msg_resp.status_code == 201
        assert "Machine:" in msg_resp.json()["content"] or mid2 in msg_resp.json()["content"]


# 5. Machine-specific question without machine
def test_copilot_machine_specific_without_machine():
    llm = LLMService(MockProvider("gemini", fails=True), MockProvider("grok", fails=True))
    res = llm.generate_grounded_response({"machine": None, "ml_findings": "", "evidence": []}, "why is this machine degrading?")
    assert "select a machine" in res["content"].lower()
    assert res["provider_used"] == "unavailable"
    assert res["fallback_level"] == 4


# 6. Gemini success
def test_gemini_success():
    gemini = MockProvider("gemini", "Gemini top-tier output")
    grok = MockProvider("grok", "Grok output")
    llm = LLMService(gemini, grok)
    res = llm.generate_grounded_response({"ml_findings": "Observed drift", "evidence": []}, "why degrading?")
    assert res["provider_used"] == "gemini"
    assert res["fallback_level"] == 0
    assert not grok.called
    assert res["content"] == "Gemini top-tier output"


# 7. Gemini failure -> Grok
def test_gemini_failure_uses_grok():
    gemini = MockProvider("gemini", fails=True)
    grok = MockProvider("grok", "Grok backup output")
    llm = LLMService(gemini, grok)
    res = llm.generate_grounded_response({"ml_findings": "Observed drift", "evidence": []}, "why degrading?")
    assert gemini.called
    assert grok.called
    assert res["provider_used"] == "grok"
    assert res["fallback_level"] == 1
    assert res["content"] == "Grok backup output"


# 8. Gemini + Grok failure -> RAG fallback
def test_gemini_grok_failure_rag_fallback():
    gemini = MockProvider("gemini", fails=True)
    grok = MockProvider("grok", fails=True)
    llm = LLMService(gemini, grok)
    context = {
        "ml_findings": "Machine: M-1. RUL: 25 cycles.",
        "evidence": [{"documentTitle": "Turbofan Manual", "section": "3.1", "page": 12, "snippet": "Check HPC stator."}],
    }
    res = llm.generate_grounded_response(context, "investigate failure")
    assert res["provider_used"] == "grounded_fallback"
    assert res["fallback_level"] == 2
    assert res["rag_used"] is True
    assert "Turbofan Manual" in res["content"]
    assert "Summary:" in res["content"]


# 9. RAG unavailable -> ML-only
def test_rag_unavailable_ml_only():
    gemini = MockProvider("gemini", fails=True)
    grok = MockProvider("grok", fails=True)
    llm = LLMService(gemini, grok)
    context = {"ml_findings": "Machine: M-1. RUL: 25 cycles. Risk: critical.", "evidence": []}
    res = llm.generate_grounded_response(context, "current risk?")
    assert res["provider_used"] == "ml_only"
    assert res["fallback_level"] == 3
    assert res["rag_used"] is False
    assert res["ml_context_used"] is True
    assert "Machine: M-1. RUL: 25 cycles." in res["content"]


# 10. ML unavailable -> unavailable
def test_ml_unavailable():
    gemini = MockProvider("gemini", fails=True)
    grok = MockProvider("grok", fails=True)
    llm = LLMService(gemini, grok)
    context = {"machine": None, "ml_findings": "", "evidence": []}
    res = llm.generate_grounded_response(context, "status?")
    assert res["provider_used"] == "unavailable"
    assert res["fallback_level"] == 4


# 11. Document upload
def test_document_upload():
    with TestClient(app) as client:
        file_content = b"Section 1.0 - Bearing Assembly\nVibration signatures exceeding 645 units indicate inner race spalling."
        resp = client.post(
            "/api/knowledge/upload",
            headers=auth_header("doc_user"),
            files={"file": ("bearing_manual.txt", file_content, "text/plain")},
            data={"category": "Manual"},
        )
        assert resp.status_code == 201
        doc_data = resp.json()
        assert doc_data["title"] == "bearing_manual"
        assert doc_data["indexedChunks"] >= 1


# 12. Document retrieval (TF-IDF search)
def test_document_retrieval_tfidf():
    with TestClient(app) as client:
        # Upload doc
        content = b"Section 2.4 - Compressor Bleed Seals\nElevated bleed enthalpy (s17) indicates high pressure air leakage past turbine cooling seals."
        client.post(
            "/api/knowledge/upload",
            headers=auth_header("search_user"),
            files={"file": ("bleed_seals.txt", content, "text/plain")},
        )
        
        # Search query
        s_resp = client.post(
            "/api/knowledge/search",
            headers=auth_header("search_user"),
            json={"query": "bleed enthalpy cooling seals", "top_k": 3},
        )
        assert s_resp.status_code == 200
        results = s_resp.json()
        assert len(results) >= 1
        assert "bleed_seals" in results[0]["document_name"]
        assert results[0]["relevance_score"] > 0


# 13. Document ownership (User isolation)
def test_document_ownership_isolation():
    with TestClient(app) as client:
        # User A uploads a private manual
        u1 = "doc_owner_a"
        u2 = "doc_intruder_b"
        content = b"Confidential Blueprint X1: Secret valve tolerances."
        upload_resp = client.post(
            "/api/knowledge/upload",
            headers=auth_header(u1),
            files={"file": ("secret_blueprint.txt", content, "text/plain")},
        )
        doc_id = upload_resp.json()["id"]

        # User B cannot access User A's document details
        assert client.get(f"/api/knowledge/documents/{doc_id}", headers=auth_header(u2)).status_code == 404

        # User B cannot access User A's chunks
        assert client.get(f"/api/knowledge/documents/{doc_id}/chunks", headers=auth_header(u2)).status_code == 404

        # User B cannot search User A's documents
        search_resp = client.post(
            "/api/knowledge/search",
            headers=auth_header(u2),
            json={"query": "Confidential Blueprint"},
        )
        assert search_resp.status_code == 200
        assert len(search_resp.json()) == 0

        # User B cannot delete User A's document
        assert client.delete(f"/api/knowledge/documents/{doc_id}", headers=auth_header(u2)).status_code == 404

        # User A can delete their document
        assert client.delete(f"/api/knowledge/documents/{doc_id}", headers=auth_header(u1)).status_code == 204
        assert client.get(f"/api/knowledge/documents/{doc_id}", headers=auth_header(u1)).status_code == 404
        assert client.get(f"/api/knowledge/documents/{doc_id}/chunks", headers=auth_header(u1)).status_code == 404


# 14. Diagnostic creation
def test_diagnostic_creation():
    with TestClient(app) as client:
        mid = create_test_machine(client, "diag_user")
        resp = client.post(
            "/api/diagnostics",
            headers=auth_header("diag_user"),
            json={"machine_id": mid},
        )
        assert resp.status_code == 201
        diag = resp.json()
        assert diag["machineId"] == mid
        assert "trigger" in diag
        assert "finding" in diag
        assert "generatorUsed" in diag
        assert "provider_used" in diag
        assert "fallback_level" in diag


# 15. Diagnostic explanation
def test_diagnostic_explanation():
    with TestClient(app) as client:
        mid = create_test_machine(client, "explain_user")
        diag = client.post("/api/diagnostics", headers=auth_header("explain_user"), json={"machine_id": mid}).json()
        
        exp_resp = client.post(f"/api/diagnostics/{diag['id']}/explain", headers=auth_header("explain_user"))
        assert exp_resp.status_code == 200
        assert "content" in exp_resp.json()
        assert "provider_used" in exp_resp.json()


# 16. Maintenance creation
def test_maintenance_creation():
    with TestClient(app) as client:
        mid = create_test_machine(client, "maint_user")
        resp = client.post(
            "/api/maintenance",
            headers=auth_header("maint_user"),
            json={
                "machine_id": mid,
                "title": "Inspect HPC Vanes",
                "description": "Exceeded thermal threshold on HPC outlet",
                "recommended_action": "Borescope inspection",
                "priority": "high",
            },
        )
        assert resp.status_code == 201
        task = resp.json()
        assert task["machineId"] == mid
        assert task["action"] == "Borescope inspection"
        assert task["priority"] == "high"


# 17. Maintenance persistence
def test_maintenance_persistence():
    with TestClient(app) as client:
        mid = create_test_machine(client, "persist_user")
        task = client.post(
            "/api/maintenance",
            headers=auth_header("persist_user"),
            json={"machine_id": mid, "title": "Calibrate Fuel Sensor", "description": "Reading offset by 3%"},
        ).json()

        # Query queue again
        queue = client.get("/api/maintenance", headers=auth_header("persist_user")).json()
        matching = [t for t in queue if t["id"] == task["id"]]
        assert len(matching) == 1
        assert matching[0]["action"] == "Calibrate Fuel Sensor"


# 18. Unauthorized machine access
def test_unauthorized_machine_access():
    with TestClient(app) as client:
        owner = "alice_owner"
        intruder = "bob_intruder"
        mid = create_test_machine(client, owner)

        # Intruder cannot read owner's machine
        assert client.get(f"/api/machines/{mid}", headers=auth_header(intruder)).status_code == 404

        # Intruder cannot create maintenance on owner's machine
        maint_bad = client.post(
            "/api/maintenance",
            headers=auth_header(intruder),
            json={"machine_id": mid, "title": "Hack", "description": "Unauthorized task"},
        )
        assert maint_bad.status_code == 404

        # Intruder cannot create diagnostic on owner's machine
        diag_bad = client.post(
            "/api/diagnostics",
            headers=auth_header(intruder),
            json={"machine_id": mid},
        )
        assert diag_bad.status_code == 404

        # Intruder cannot link owner's machine to copilot conversation
        convo_bad = client.post(
            "/api/copilot/conversations",
            headers=auth_header(intruder),
            json={"machine_id": mid},
        )
        assert convo_bad.status_code == 404
