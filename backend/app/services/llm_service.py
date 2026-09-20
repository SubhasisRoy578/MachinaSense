"""Grounded provider chain for engineering assistance.

Provider calls live here so routes and diagnostics use exactly the same policy.
"""
from __future__ import annotations

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
            params={"key": settings.GEMINI_API_KEY}, json=payload, timeout=20,
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
            json={"model": settings.GROK_MODEL, "messages": [{"role": "user", "content": _prompt(context, question)}], "temperature": 0.2},
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
    return f"""You are MachinaSense Engineering Copilot. Answer only from the supplied structured context.
Never invent sensor values, RUL, anomaly scores, documents, citations, machine history, or model metrics.
Clearly distinguish Observed Data, Model Finding, Document Evidence, Interpretation, and Recommended Investigation.
Recommendations are investigation guidance, not guaranteed physical maintenance instructions.

QUESTION: {question}
STRUCTURED ENGINEERING CONTEXT: {context}
"""


class LLMService:
    def __init__(self, gemini: Any = None, grok: Any = None):
        self.gemini = gemini or GeminiProvider()
        self.grok = grok or GrokProvider()

    def generate_grounded_response(self, context: Dict[str, Any], question: str) -> Dict[str, Any]:
        for provider, model in ((self.gemini, settings.GEMINI_MODEL), (self.grok, settings.GROK_MODEL)):
            try:
                return {"content": provider.generate(context, question), "provider_used": provider.name,
                        "fallback_level": 0 if provider.name == "gemini" else 1,
                        "rag_used": bool(context.get("evidence")), "ml_context_used": bool(context.get("ml_findings")), "model_name": model}
            except ProviderError:
                continue
            except requests.RequestException:
                continue
        evidence = context.get("evidence") or []
        if evidence:
            citations = "; ".join(f"{item['documentTitle']} ({item['section']}, p.{item['page']})" for item in evidence)
            return {"content": f"Grounded deterministic response\n\nMachine Evidence: {context.get('ml_findings')}\n\nTechnical Evidence: {citations}\n\nEvidence Status: Documentation was retrieved; investigate the observed model findings before taking action.", "provider_used": "grounded_fallback", "fallback_level": 2, "rag_used": True, "ml_context_used": bool(context.get("ml_findings")), "model_name": "deterministic-rag"}
        if context.get("ml_findings"):
            return {"content": f"ML-only factual summary\n\n{context['ml_findings']}\n\nTechnical evidence: Unavailable.", "provider_used": "ml_only", "fallback_level": 3, "rag_used": False, "ml_context_used": True, "model_name": "ml-artifacts"}
        return {"content": "Insufficient machine data is available to provide an analysis.", "provider_used": "unavailable", "fallback_level": 4, "rag_used": False, "ml_context_used": False, "model_name": "unavailable"}


llm_service = LLMService()
