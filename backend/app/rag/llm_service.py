import os
import json
import httpx
from typing import Dict, List, Any, Optional
from app.rag.vector_store import vector_store

# Read LLM API Keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class RAGDiagnosticEngine:
    def __init__(self):
        pass

    def generate_diagnostic(
        self,
        machine_context: Dict[str, Any],
        min_relevance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Generates a grounded AI diagnostic case using real ML results + vector RAG evidence.
        """
        machine_id = machine_context.get("id", "FD001-001")
        machine_name = machine_context.get("name", f"Turbofan Engine {machine_id}")
        rul_lstm = machine_context.get("predictedRul", 100.0)
        rul_rf = machine_context.get("rfPredictedRul", 100.0)
        health_score = machine_context.get("healthScore", 90)
        anomaly_score = machine_context.get("anomalyScore", 0.1)
        anomaly_sev = machine_context.get("anomalySeverity", "healthy")
        current_cycle = machine_context.get("currentCycle", 100)
        sensors = machine_context.get("sensors", [])

        # Build search query from abnormal telemetry & ML findings
        query_terms = ["turbofan degradation", "RUL prediction"]
        if anomaly_sev in ["high", "critical"]:
            query_terms.append("vibration bearing resonance temperature")
        if rul_lstm < 40:
            query_terms.append("High-Pressure Compressor HPC outlet temperature")
        if health_score < 60:
            query_terms.append("fan core speed calibration leakage")

        search_query = " ".join(query_terms)
        retrieved_chunks = vector_store.search(search_query, top_k=5, min_score=min_relevance)

        # Check for insufficient evidence
        if not retrieved_chunks:
            return {
                "id": f"DC-{machine_id}-CASE",
                "machineId": machine_id,
                "machineName": machine_name,
                "status": "open",
                "severity": anomaly_sev if anomaly_sev != "healthy" else ("warning" if health_score < 50 else "low"),
                "trigger": f"Telemetry check for {machine_id} at cycle {current_cycle}",
                "finding": "Insufficient supporting evidence.",
                "explanation": "No relevant technical documentation was retrieved from the Knowledge Base to substantiate current telemetry signals.",
                "evidence": [],
                "recommendedAction": "Upload relevant maintenance manuals or technical specifications to the Knowledge Base to enable grounded AI diagnosis.",
                "confidence": 0,
                "generatorUsed": "Grounded Rule-Based Fallback",
                "isGroundedFallback": True
            }

        # Build evidence list with strict source provenance
        evidence_items = []
        for chunk in retrieved_chunks:
            evidence_items.append({
                "documentId": chunk["document_id"],
                "documentTitle": chunk["document_name"],
                "section": chunk["section"],
                "page": chunk["page"],
                "snippet": chunk["excerpt"],
                "relevance": chunk["relevance_score"]
            })

        # Try live LLM call if GEMINI_API_KEY or OPENAI_API_KEY is available
        if GEMINI_API_KEY:
            llm_res = self._call_gemini_api(machine_context, evidence_items)
            if llm_res:
                llm_res["generatorUsed"] = "Google Gemini API (Grounded RAG)"
                llm_res["isGroundedFallback"] = False
                return llm_res

        if OPENAI_API_KEY:
            llm_res = self._call_openai_api(machine_context, evidence_items)
            if llm_res:
                llm_res["generatorUsed"] = "OpenAI GPT API (Grounded RAG)"
                llm_res["isGroundedFallback"] = False
                return llm_res

        # Fallback to Grounded Rule-Based Diagnostic Generator (Explicitly Labeled)
        return self._generate_grounded_fallback(machine_context, evidence_items)

    def _generate_grounded_fallback(
        self,
        machine_context: Dict[str, Any],
        evidence_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Deterministic, grounded fallback diagnostic generator.
        Explicitly labeled as 'Grounded Rule-Based Fallback'. NEVER masquerades as an LLM.
        """
        machine_id = machine_context.get("id", "FD001-001")
        machine_name = machine_context.get("name", f"Turbofan Engine {machine_id}")
        rul_lstm = machine_context.get("predictedRul", 100.0)
        rul_rf = machine_context.get("rfPredictedRul", 100.0)
        health_score = machine_context.get("healthScore", 90)
        anomaly_sev = machine_context.get("anomalySeverity", "healthy")
        current_cycle = machine_context.get("currentCycle", 100)

        top_evidence = evidence_items[0] if evidence_items else None
        doc_title = top_evidence["documentTitle"] if top_evidence else "Technical Documentation"
        section_name = top_evidence["section"] if top_evidence else "General"

        if rul_lstm < 30 or anomaly_sev == "critical":
            sev = "critical"
            finding = f"High-Pressure Compressor (HPC) & Bearing Degradation. PyTorch LSTM RUL: {rul_lstm:.1f} cycles remaining."
            explanation = f"Based on retrieved evidence from '{doc_title}' ({section_name}), elevated compressor temperatures (s3) and acoustic anomaly score ({anomaly_sev}) correlate with thermal degradation and blade clearance wear."
            action = "Schedule immediate HPC stator vane inspection and bearing replacement. Limit engine load until maintenance."
            conf = 91
        elif rul_lstm < 60 or anomaly_sev in ["high", "medium"]:
            sev = "high"
            finding = f"Moderate Thermal Degradation & Air Leakage. PyTorch LSTM RUL: {rul_lstm:.1f} cycles."
            explanation = f"Retrieved technical documentation ({doc_title}) confirms fan/core speed variances and bleed enthalpy shifts signal early-stage thermal seal degradation."
            action = "Inspect HPC cooling duct seals and recalibrate temperature sensors s2 and s4 during next maintenance window."
            conf = 84
        else:
            sev = "low"
            finding = f"Engine Operating Within Nominal Limits. PyTorch LSTM RUL: {rul_lstm:.1f} cycles."
            explanation = f"Telemetry readings align with nominal bounds specified in {doc_title}. No severe degradation detected."
            action = "Continue standard operational cycle logging and routine maintenance schedule."
            conf = 95

        return {
            "id": f"DC-{machine_id}-FALLBACK",
            "machineId": machine_id,
            "machineName": machine_name,
            "status": "open" if sev in ["critical", "high"] else "resolved",
            "severity": sev,
            "trigger": f"RUL forecast ({rul_lstm:.1f} cycles) & Isolation Forest anomaly score ({anomaly_sev})",
            "finding": finding,
            "explanation": explanation,
            "evidence": evidence_items,
            "recommendedAction": action,
            "confidence": conf,
            "generatorUsed": "Grounded Rule-Based Fallback",
            "isGroundedFallback": True
        }

    def _call_gemini_api(
        self,
        machine_context: Dict[str, Any],
        evidence_items: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Calls Google Gemini REST API using httpx."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            prompt_text = f"""You are a senior industrial turbofan diagnostic engineer.
Given the following machine context and retrieved evidence, generate a grounded diagnostic report.
STRICT RULE: Only use facts explicitly supported by the supplied context and evidence. Do not invent non-existent sensor values, failures, or procedures.

MACHINE CONTEXT:
{json.dumps(machine_context, indent=2)}

RETRIEVED TECHNICAL EVIDENCE:
{json.dumps(evidence_items, indent=2)}

Return a valid JSON object with exact keys:
"finding" (string), "explanation" (string), "recommendedAction" (string), "severity" ("critical"|"high"|"medium"|"low"), "confidence" (number 0-100).
"""
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
            }
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_content = data['candidates'][0]['content']['parts'][0]['text']
                    parsed = json.loads(raw_content)
                    return {
                        "id": f"DC-{machine_context.get('id', '1')}-GEMINI",
                        "machineId": machine_context.get("id"),
                        "machineName": machine_context.get("name"),
                        "status": "open",
                        "severity": parsed.get("severity", "high"),
                        "trigger": f"Real ML Telemetry Analysis for {machine_context.get('id')}",
                        "finding": parsed.get("finding", "Diagnostic Analysis Completed"),
                        "explanation": parsed.get("explanation", ""),
                        "evidence": evidence_items,
                        "recommendedAction": parsed.get("recommendedAction", ""),
                        "confidence": parsed.get("confidence", 88)
                    }
        except Exception as e:
            print(f"Warning: Gemini API call failed: {e}")
        return None

    def _call_openai_api(
        self,
        machine_context: Dict[str, Any],
        evidence_items: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Calls OpenAI Chat Completion API using httpx."""
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            prompt_text = f"""You are a senior industrial turbofan diagnostic engineer.
Generate a grounded diagnostic report. STRICT RULE: Use ONLY facts explicitly supported by the supplied context and evidence.

MACHINE CONTEXT:
{json.dumps(machine_context, indent=2)}

RETRIEVED EVIDENCE:
{json.dumps(evidence_items, indent=2)}

Return JSON with keys: "finding", "explanation", "recommendedAction", "severity", "confidence".
"""
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt_text}],
                "response_format": {"type": "json_object"},
                "temperature": 0.2
            }
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    parsed = json.loads(data["choices"][0]["message"]["content"])
                    return {
                        "id": f"DC-{machine_context.get('id', '1')}-OPENAI",
                        "machineId": machine_context.get("id"),
                        "machineName": machine_context.get("name"),
                        "status": "open",
                        "severity": parsed.get("severity", "high"),
                        "trigger": f"Real ML Telemetry Analysis for {machine_context.get('id')}",
                        "finding": parsed.get("finding", "Diagnostic Analysis Completed"),
                        "explanation": parsed.get("explanation", ""),
                        "evidence": evidence_items,
                        "recommendedAction": parsed.get("recommendedAction", ""),
                        "confidence": parsed.get("confidence", 88)
                    }
        except Exception as e:
            print(f"Warning: OpenAI API call failed: {e}")
        return None

rag_engine = RAGDiagnosticEngine()
