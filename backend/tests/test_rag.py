import os, sys
BASE_DIR=os.path.normpath(os.path.join(os.path.dirname(__file__),".."));sys.path.insert(0,BASE_DIR)
from app.services.llm_service import LLMService, ProviderError

class Provider:
    def __init__(self,name,result=None): self.name=name;self.result=result;self.called=False
    def generate(self,context,question):
        self.called=True
        if isinstance(self.result,Exception): raise self.result
        return self.result

ML={"ml_findings":"RUL: 28 cycles. Risk: High. Anomaly: Detected.","evidence":[]}
EVIDENCE={**ML,"evidence":[{"documentTitle":"Manual","section":"2.1","page":3}]}
def test_gemini_success_does_not_call_grok():
    gemini=Provider("gemini","Gemini response");grok=Provider("grok","Grok response")
    result=LLMService(gemini,grok).generate_grounded_response(ML,"status?")
    assert result["provider_used"]=="gemini" and not grok.called
def test_gemini_failure_uses_grok():
    result=LLMService(Provider("gemini",ProviderError("failed")),Provider("grok","Grok response")).generate_grounded_response(ML,"status?")
    assert result["provider_used"]=="grok" and result["fallback_level"]==1
def test_both_provider_failures_use_grounded_evidence():
    result=LLMService(Provider("gemini",ProviderError("failed")),Provider("grok",ProviderError("failed"))).generate_grounded_response(EVIDENCE,"status?")
    assert result["provider_used"]=="grounded_fallback" and result["rag_used"] is True
def test_no_rag_uses_ml_only():
    result=LLMService(Provider("gemini",ProviderError("failed")),Provider("grok",ProviderError("failed"))).generate_grounded_response(ML,"status?")
    assert result["provider_used"]=="ml_only"
def test_no_context_is_unavailable():
    result=LLMService(Provider("gemini",ProviderError("failed")),Provider("grok",ProviderError("failed"))).generate_grounded_response({"ml_findings":"","evidence":[]},"status?")
    assert result["provider_used"]=="unavailable"
