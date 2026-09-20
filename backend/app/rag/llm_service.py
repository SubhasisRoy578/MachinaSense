"""Compatibility diagnostic engine using deterministic, provenance-aware RAG.

Live provider selection belongs to ``app.services.llm_service``. This module
intentionally never calls a third-party hosted model; provider routing is server-side.
"""
from typing import Any, Dict
from app.rag.vector_store import vector_store

class RAGDiagnosticEngine:
    def generate_diagnostic(self, machine_context: Dict[str, Any], min_relevance: float = .05) -> Dict[str, Any]:
        machine_id=machine_context.get("id", "unknown")
        findings=f"RUL: {machine_context.get('predictedRul', 'unavailable')} cycles; anomaly: {machine_context.get('anomalySeverity', 'unavailable')}."
        evidence=vector_store.search("maintenance RUL anomaly",top_k=5,min_score=min_relevance)
        sources=[{"documentId":x["document_id"],"documentTitle":x["document_name"],"section":x["section"],"page":x["page"],"snippet":x["excerpt"],"relevance":x["relevance_score"]} for x in evidence]
        return {"id":f"diagnostic-{machine_id}","machineId":machine_id,"machineName":machine_context.get("name",machine_id),"status":"open","severity":machine_context.get("anomalySeverity","low"),"trigger":"ML/RAG investigation","finding":findings,"explanation":"Grounded deterministic fallback based only on supplied ML context and retrieved evidence.","evidence":sources,"recommendedAction":"Review the observations and cited documentation before maintenance action.","confidence":0 if not sources else 80,"generatorUsed":"Grounded Deterministic Fallback","isGroundedFallback":True,"provider_used":"grounded_fallback","fallback_level":2}
rag_engine=RAGDiagnosticEngine()
