"""Grounded provider chain for engineering assistance.

Provider calls live here so routes and diagnostics use exactly the same policy.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List
import requests

from app.config import settings


class ProviderError(RuntimeError):
    pass


class GeminiProvider:
    name = "gemini"

    def generate(self, context: Dict[str, Any], question: str) -> str:
        if not settings.GEMINI_API_KEY:
            raise ProviderError("Gemini is not configured")
        payload = {"contents": [{"role": "user", "parts": [{"text": _prompt(context, question)}]}]}
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent",
            params={"key": settings.GEMINI_API_KEY},
            json=payload,
            timeout=20,
        )
        if not response.ok:
            raise ProviderError(f"Gemini returned {response.status_code}")
        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("Gemini returned a malformed response") from exc
        if not text:
            raise ProviderError("Gemini returned an empty response")
        return text


class GrokProvider:
    name = "grok"

    def generate(self, context: Dict[str, Any], question: str) -> str:
        if not settings.GROK_API_KEY:
            raise ProviderError("Grok is not configured")
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.GROK_API_KEY}"},
            json={
                "model": settings.GROK_MODEL,
                "messages": [{"role": "user", "content": _prompt(context, question)}],
                "temperature": 0.2,
            },
            timeout=20,
        )
        if not response.ok:
            raise ProviderError(f"Grok returned {response.status_code}")
        try:
            text = response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("Grok returned a malformed response") from exc
        if not text:
            raise ProviderError("Grok returned an empty response")
        return text


def _prompt(context: Dict[str, Any], question: str) -> str:
    return f"""You are MachinaSense Engineering Copilot, an industrial predictive maintenance assistant.
Answer authoritatively, clearly, and concisely.

CRITICAL OPERATIONAL RULES:
1. Never invent or hallucinate sensor values, RUL numbers, anomaly scores, machine IDs, maintenance records, or technical document citations.
2. If machine context is provided, ground your analysis strictly in the supplied structured engineering context.
   Where applicable for machine diagnostic investigations, organize your response with the following sections:
   - Summary
   - Machine Evidence
   - ML Findings
   - Technical Evidence
   - Interpretation
   - Recommended Investigation
   - Evidence Status
   Recommendations are investigative guidance, not guaranteed physical maintenance instructions.
3. If the user asks a machine-specific inquiry (such as machine RUL, degradation cause, current anomaly, or maintenance priority) but NO machine is selected in context, do NOT fabricate data. Explicitly instruct the user to select a machine from the machine selector.
4. If the user asks a general engineering question, greeting, or concept inquiry (e.g. hello, what is predictive maintenance, explain RUL, explain anomaly detection, what can you do), answer directly and informatively using standard industrial predictive maintenance principles, without requiring machine telemetry.

QUESTION: {question}
STRUCTURED ENGINEERING CONTEXT: {context}
"""


def _is_machine_specific(question: str) -> bool:
    q = question.lower().strip()
    triggers = [
        "this machine", "my machine", "the machine", "selected machine",
        "current machine", "this asset", "this engine", "its rul", "its health",
        "its anomaly", "its degradation", "why is it", "why is this", "what is its",
        "why is machine", "what is this machine", "why is this machine degrading",
        "what is this machine's rul", "why is this machine high risk",
        "what anomalies were detected", "what maintenance should be investigated",
        "status?"
    ]
    return any(t in q for t in triggers)


def _is_general_query(question: str) -> bool:
    q = question.lower().strip()
    # Strip basic punctuation
    cleaned = re.sub(r"[^\w\s]", "", q).strip()
    greetings = {"hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"}
    if cleaned in greetings:
        return True
    
    general_phrases = [
        "what can you do", "help", "capabilities", "who are you",
        "what is predictive maintenance", "predictive maintenance",
        "explain rul", "explain remaining useful life", "what is rul", "what is remaining useful life",
        "explain anomaly detection", "what is anomaly detection", "how does anomaly detection work",
        "how does rul work", "explain isolation forest", "c-mapss", "turbofan", "sensors"
    ]
    return any(p in q for p in general_phrases)


def _deterministic_general_response(question: str) -> str:
    q = question.lower().strip()
    cleaned = re.sub(r"[^\w\s]", "", q).strip()
    greetings = {"hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"}

    if cleaned in greetings:
        return (
            "Hello! I am MachinaSense Engineering Copilot, your industrial predictive maintenance assistant.\n\n"
            "I can help you with:\n"
            "• Predictive maintenance concepts and workflows\n"
            "• Remaining Useful Life (RUL) modeling (PyTorch LSTM & Random Forest)\n"
            "• Anomaly detection analysis (Isolation Forest)\n"
            "• Telemetry interpretation and degradation curve tracking\n"
            "• Grounded technical document retrieval\n\n"
            "Select an asset from the machine selector above to begin machine-specific diagnostics, "
            "or ask any general engineering question."
        )

    if any(p in q for p in ["what can you do", "help", "capabilities", "who are you"]):
        return (
            "I am MachinaSense Engineering Copilot. Here is what I can do:\n\n"
            "1. Domain Explanations: Explain predictive maintenance, RUL estimation, sensor signatures, and anomaly detection.\n"
            "2. Telemetry & Sensor Monitoring: Evaluate real-time readings (temperatures, pressures, fan and core speeds, vibration) against baselines.\n"
            "3. RUL Forecasting: Synthesize predictions from the PyTorch LSTM temporal model and Random Forest baseline with confidence intervals.\n"
            "4. Anomaly Detection: Analyze Isolation Forest anomaly scores to identify multivariate operational deviations.\n"
            "5. Grounded RAG Documentation: Correlate telemetry with your uploaded technical manuals, providing citations (document title, section, page).\n"
            "6. Maintenance Prioritization: Recommend targeted inspection priorities for open maintenance tickets."
        )

    if "predictive maintenance" in q:
        return (
            "Predictive Maintenance (PdM) is a condition-driven maintenance strategy that continuously evaluates equipment health "
            "using sensor telemetry, statistical modeling, and machine learning to forecast component degradation and schedule interventions before functional failure.\n\n"
            "Core components in MachinaSense:\n"
            "• Condition Monitoring: Streaming multi-sensor readings (vibration, temperatures, pressures, rotational speeds).\n"
            "• Anomaly Detection: Unsupervised Isolation Forest flags early abnormal excursions.\n"
            "• Prognostics: Deep learning (PyTorch LSTM) models time-series sensor sequences to forecast Remaining Useful Life (RUL).\n"
            "• Evidence-Based Guidance: Grounding recommendations in technical manuals to verify failure modes prior to physical maintenance."
        )

    if any(p in q for p in ["explain rul", "explain remaining useful life", "what is rul", "what is remaining useful life"]):
        return (
            "Remaining Useful Life (RUL) is the estimated operational time or cycle count remaining before an asset or component degrades below acceptable operating thresholds.\n\n"
            "In MachinaSense:\n"
            "• Sequence Modeling: A 2-layer PyTorch LSTM with a 30-cycle sliding window captures temporal degradation trajectories across 15 operational sensors.\n"
            "• Baseline Regression: A Random Forest regressor provides benchmark comparison on cycle-level features.\n"
            "• Standardized RUL Cap: Set at 125 cycles following the NASA C-MAPSS FD001 benchmark standard to prevent overestimating healthy asset life.\n"
            "• Decision Thresholds: RUL < 70 cycles indicates elevated warning; RUL < 30 cycles indicates critical maintenance urgency."
        )

    if any(p in q for p in ["explain anomaly detection", "what is anomaly detection", "how does anomaly detection work"]):
        return (
            "Anomaly detection in industrial operations identifies sensor patterns and operating states that deviate significantly from established baseline behavior.\n\n"
            "In MachinaSense:\n"
            "• Model: An unsupervised Isolation Forest algorithm fitted across 15 normalized sensor features.\n"
            "• Scoring: Computes anomaly scores based on the isolation path length of multivariate samples.\n"
            "• Severity Classification: Categorizes asset health into healthy, medium, high, and critical severity tiers.\n"
            "• Early Warning: Catches abnormal thermal, pressure, or vibrational excursions prior to catastrophic functional breakdown."
        )

    return (
        "MachinaSense Engineering Copilot is ready to assist. You can ask general predictive maintenance questions "
        "(such as explaining RUL, anomaly detection, or sensor baselines), or select a machine from the machine selector "
        "to evaluate real-time telemetry, model forecasts, and grounded diagnostic investigations."
    )


class LLMService:
    def __init__(self, gemini: Any = None, grok: Any = None):
        self.gemini = gemini or GeminiProvider()
        self.grok = grok or GrokProvider()

    def generate_grounded_response(self, context: Dict[str, Any], question: str) -> Dict[str, Any]:
        # 1. Attempt Primary (Gemini) then Secondary (Grok)
        for provider, model in ((self.gemini, settings.GEMINI_MODEL), (self.grok, settings.GROK_MODEL)):
            try:
                return {
                    "content": provider.generate(context, question),
                    "provider_used": provider.name,
                    "fallback_level": 0 if provider.name == "gemini" else 1,
                    "rag_used": bool(context.get("evidence")),
                    "ml_context_used": bool(context.get("ml_findings")),
                    "model_name": model,
                }
            except ProviderError:
                continue
            except requests.RequestException:
                continue

        evidence = context.get("evidence") or []
        has_ml = bool(context.get("ml_findings"))
        has_machine = bool(context.get("machine"))

        # 2. Check if this is a machine-specific query without a machine selected
        if _is_machine_specific(question) and not has_machine and not has_ml:
            return {
                "content": (
                    "To analyze machine health, remaining useful life (RUL), anomaly severity, or maintenance investigations "
                    "for a specific asset, please select a machine from the machine selector above. "
                    "Once selected, I will incorporate its real-time telemetry, model predictions, and diagnostic evidence into our session."
                ),
                "provider_used": "unavailable",
                "fallback_level": 4,
                "rag_used": False,
                "ml_context_used": False,
                "model_name": "unavailable",
            }

        # 3. Deterministic Grounded RAG Fallback (fallback_level = 2)
        if evidence:
            citations = "; ".join(f"{item['documentTitle']} ({item['section']}, p.{item['page']})" for item in evidence)
            if has_ml:
                content = (
                    f"Summary: Grounded maintenance investigation based on live telemetry and technical documentation.\n\n"
                    f"Machine Evidence:\n{context.get('ml_findings')}\n\n"
                    f"Technical Evidence:\n{citations}\n\n"
                    f"Interpretation:\nTelemetry observations correlate with documented failure modes and thresholds in the technical library.\n\n"
                    f"Recommended Investigation:\nReview the cited manuals and cross-reference observed sensor deviations before field intervention.\n\n"
                    f"Evidence Status:\nGrounded technical documentation was retrieved from the library."
                )
            else:
                content = (
                    f"Technical Evidence:\n{citations}\n\n"
                    f"Summary:\nBased on retrieved engineering documentation: {evidence[0].get('snippet', '')}\n\n"
                    f"Evidence Status:\nRetrieved from indexed knowledge base documents."
                )
            return {
                "content": content,
                "provider_used": "grounded_fallback",
                "fallback_level": 2,
                "rag_used": True,
                "ml_context_used": has_ml,
                "model_name": "deterministic-rag",
            }

        # 4. ML-only Fallback (fallback_level = 3)
        if has_ml:
            content = (
                f"Summary: Live telemetry and predictive model summary.\n\n"
                f"Machine Evidence:\n{context['ml_findings']}\n\n"
                f"ML Findings:\nPyTorch LSTM and Random Forest models evaluated available sensor telemetry.\n\n"
                f"Technical Evidence:\nUnavailable. No technical manuals have been indexed for this asset.\n\n"
                f"Interpretation:\nOperating health and risk are derived from current cycle telemetry.\n\n"
                f"Recommended Investigation:\nInspect sensor trend deviations and upload relevant equipment manuals for grounded diagnostic synthesis.\n\n"
                f"Evidence Status:\nTelemetry only; technical library evidence unavailable."
            )
            return {
                "content": content,
                "provider_used": "ml_only",
                "fallback_level": 3,
                "rag_used": False,
                "ml_context_used": True,
                "model_name": "ml-artifacts",
            }

        # 5. General question / capability inquiry without machine or documents
        if _is_general_query(question):
            return {
                "content": _deterministic_general_response(question),
                "provider_used": "grounded_fallback",
                "fallback_level": 2,
                "rag_used": False,
                "ml_context_used": False,
                "model_name": "deterministic-domain",
            }

        # 6. Data unavailable (fallback_level = 4)
        return {
            "content": "Insufficient machine data is available to provide an analysis. Please select a machine or provide operational context.",
            "provider_used": "unavailable",
            "fallback_level": 4,
            "rag_used": False,
            "ml_context_used": False,
            "model_name": "unavailable",
        }


llm_service = LLMService()
