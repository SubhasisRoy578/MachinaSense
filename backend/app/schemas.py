from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str
    backend: str = "live"
    models_loaded: bool
    rag_status: str = "ready"  # 'ready', 'unindexed', 'error'
    llm_status: str = "fallback"  # 'gemini', 'openai', 'fallback'
    dataset: str = "C-MAPSS FD001"
    models: List[str]
    artifacts_path: str

class PredictRULRequest(BaseModel):
    features: Optional[List[float]] = Field(None, description="Array of 15 sensor readings (s2, s3, s4, s6, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21)")
    sequence: Optional[List[List[float]]] = Field(None, description="30x15 time series sequence matrix")
    use_lstm: bool = True

class PredictRULResponse(BaseModel):
    predicted_rul: float
    rf_predicted_rul: float
    model_used: str
    confidence_interval: Dict[str, float]

class DetectAnomalyRequest(BaseModel):
    features: List[float] = Field(..., description="Array of 15 sensor readings")

class DetectAnomalyResponse(BaseModel):
    anomaly_score: float
    severity: str
    status: str
    is_anomaly: bool

class MachineSummary(BaseModel):
    id: str
    numericId: int
    name: str
    model: str
    location: str
    status: str
    runningHours: int
    currentCycle: int
    healthScore: int
    predictedRul: float
    rfPredictedRul: float
    anomalySeverity: str
    anomalyScore: float
    lastMaintenance: str

class AnalyticsModelsResponse(BaseModel):
    dataset: str
    metadata: Dict[str, Any]
    evaluation: Dict[str, Any]

# Phase 5 Knowledge & Diagnostic Schemas
class KnowledgeDocumentResponse(BaseModel):
    id: str
    title: str
    type: str
    size: int
    uploadDate: str
    processingStatus: str
    indexedChunks: int
    sourceCategory: str

class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.05

class KnowledgeSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    section: str
    excerpt: str
    relevance_score: float

class CreateDiagnosticRequest(BaseModel):
    machine_id: str
    use_llm: Optional[bool] = True

class EvidenceItem(BaseModel):
    documentId: str
    documentTitle: str
    section: str
    page: int
    snippet: str
    relevance: float

class DiagnosticResponse(BaseModel):
    id: str
    machineId: str
    machineName: str
    status: str
    severity: str
    trigger: str
    finding: str
    explanation: str
    evidence: List[EvidenceItem]
    recommendedAction: str
    confidence: int
    generatorUsed: str
    isGroundedFallback: bool
