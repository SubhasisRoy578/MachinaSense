import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app
from app.rag.extractor import extract_text, DocumentExtractionError
from app.rag.chunker import chunk_pages
from app.rag.vector_store import vector_store
from app.rag.llm_service import rag_engine

def test_text_extraction_txt():
    content = b"Section 1.0 - Introduction\n\nTurbofan engine degradation modeling."
    pages, full_text = extract_text("manual.txt", content)
    assert len(pages) > 0
    assert "Section 1.0" in full_text

def test_invalid_file_type():
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_text("script.py", b"print('hello')")
    assert "Unsupported file extension" in str(exc_info.value)

def test_empty_file_validation():
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_text("empty.txt", b"")
    assert "empty" in str(exc_info.value).lower()

def test_chunker_provenance():
    pages = [
        {"page": 1, "section": "Section 2.1 - Compressor", "text": "High pressure compressor temperature s3 exceeds limits."}
    ]
    chunks = chunk_pages("DOC-101", "Compressor Manual", pages)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["document_id"] == "DOC-101"
    assert c["document_name"] == "Compressor Manual"
    assert c["page"] == 1
    assert c["section"] == "Section 2.1 - Compressor"
    assert "chunk_id" in c

def test_vector_search():
    results = vector_store.search("compressor temperature s3", top_k=3, min_score=0.01)
    assert len(results) > 0
    r0 = results[0]
    assert "document_id" in r0
    assert "relevance_score" in r0
    assert "excerpt" in r0

def test_rag_diagnostic_generation():
    machine_context = {
        "id": "FD001-001",
        "name": "Turbofan Engine FD001-001",
        "predictedRul": 25.5,
        "rfPredictedRul": 32.0,
        "healthScore": 20,
        "anomalyScore": 0.85,
        "anomalySeverity": "critical",
        "currentCycle": 180,
        "sensors": []
    }
    res = rag_engine.generate_diagnostic(machine_context)
    assert "finding" in res
    assert "explanation" in res
    assert "evidence" in res
    assert "generatorUsed" in res
    assert res["isGroundedFallback"] is True  # No LLM API key in test environment

def test_insufficient_evidence():
    machine_context = {
        "id": "FD001-099",
        "name": "Engine 99",
        "predictedRul": 100.0,
        "rfPredictedRul": 100.0,
        "healthScore": 95,
        "anomalyScore": 0.0,
        "anomalySeverity": "healthy",
        "currentCycle": 10
    }
    # Pass impossibly high min_relevance threshold to trigger insufficient evidence
    res = rag_engine.generate_diagnostic(machine_context, min_relevance=0.999)
    assert res["finding"] == "Insufficient supporting evidence."
    assert len(res["evidence"]) == 0

def test_fastapi_knowledge_endpoints():
    with TestClient(app) as client:
        # GET documents
        docs_res = client.get("/api/knowledge/documents")
        assert docs_res.status_code == 200
        docs = docs_res.json()
        assert len(docs) > 0

        # Search knowledge base
        search_res = client.post("/api/knowledge/search", json={"query": "compressor temperature s3"})
        assert search_res.status_code == 200
        results = search_res.json()
        assert len(results) > 0

        # Upload valid TXT file
        upload_res = client.post(
            "/api/knowledge/upload",
            files={"file": ("test_manual.txt", b"Section 1.1 - Bleed Valve Maintenance\n\nCheck bleed valves for contamination.", "text/plain")},
            data={"category": "Manual"}
        )
        assert upload_res.status_code == 200
        uploaded_doc = upload_res.json()
        assert uploaded_doc["processingStatus"] == "ready"

def test_fastapi_diagnostic_endpoints():
    with TestClient(app) as client:
        # Create diagnostic case
        create_res = client.post("/api/diagnostics", json={"machine_id": "FD001-001"})
        assert create_res.status_code == 200
        diag = create_res.json()
        assert diag["machineId"] == "FD001-001"
        assert "evidence" in diag

        # List diagnostic cases
        list_res = client.get("/api/diagnostics")
        assert list_res.status_code == 200
        cases = list_res.json()
        assert len(cases) > 0
