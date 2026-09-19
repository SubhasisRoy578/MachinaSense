import os
import sys
import json
import logging
import re
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.config import settings
from app.ml_service import ml_service, FEATURE_COLS
from app.rag.vector_store import vector_store
from app.rag.extractor import extract_text, DocumentExtractionError
from app.rag.chunker import chunk_pages
from app.rag.llm_service import rag_engine, GEMINI_API_KEY, OPENAI_API_KEY
from app.schemas import (
    HealthResponse, PredictRULRequest, PredictRULResponse,
    DetectAnomalyRequest, DetectAnomalyResponse, MachineSummary,
    AnalyticsModelsResponse, KnowledgeDocumentResponse, KnowledgeSearchRequest,
    KnowledgeSearchResult, CreateDiagnosticRequest, DiagnosticResponse
)

# Setup structured logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("machinasense")

CASES_FILE = os.path.join(BASE_DIR, "artifacts", "knowledge", "diagnostic_cases.json")
diagnostic_cases_cache: Dict[str, Dict[str, Any]] = {}

def sanitize_filename(filename: str) -> str:
    """Sanitizes filename removing directory traversal and dangerous characters."""
    base = os.path.basename(filename)
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    return sanitized or "document.txt"

def load_diagnostic_cases():
    global diagnostic_cases_cache
    if os.path.exists(CASES_FILE):
        try:
            with open(CASES_FILE, "r", encoding="utf-8") as f:
                cases_list = json.load(f)
                diagnostic_cases_cache = {c["id"]: c for c in cases_list}
            logger.info(f"Loaded {len(diagnostic_cases_cache)} persistent diagnostic cases.")
        except Exception as e:
            logger.warning(f"Warning loading diagnostic cases: {e}")

def save_diagnostic_cases():
    try:
        os.makedirs(os.path.dirname(CASES_FILE), exist_ok=True)
        with open(CASES_FILE, "w", encoding="utf-8") as f:
            json.dump(list(diagnostic_cases_cache.values()), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save diagnostic cases: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [Env: {settings.ENVIRONMENT}]...")
    ml_service.load_artifacts()
    load_diagnostic_cases()
    yield
    logger.info("Shutting down MachinaSense FastAPI Backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-Hardened Real ML + Grounded RAG Industrial Diagnostics API",
    version=settings.VERSION,
    lifespan=lifespan
)

# Safe CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global safe exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."}
    )

# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def get_health():
    models_ready = ml_service.is_loaded
    rag_ready = vector_store.is_indexed and len(vector_store.documents) > 0
    
    if GEMINI_API_KEY:
        llm_mode = "gemini"
    elif OPENAI_API_KEY:
        llm_mode = "openai"
    else:
        llm_mode = "fallback"

    return HealthResponse(
        status="healthy" if (models_ready and rag_ready) else ("degraded" if models_ready else "error"),
        backend="live",
        models_loaded=models_ready,
        rag_status="ready" if rag_ready else "unindexed",
        llm_status=llm_mode,
        dataset="C-MAPSS FD001",
        models=[
            "Random Forest Regressor Baseline",
            "PyTorch LSTM Temporal RUL",
            "Isolation Forest Anomaly Detector",
            "TF-IDF Vector RAG Engine"
        ],
        artifacts_path=os.path.join(BASE_DIR, "artifacts")
    )

# ---------------------------------------------------------
# Phase 4 Machine Endpoints (PRESERVED 100%)
# ---------------------------------------------------------
@app.get("/api/machines", response_model=List[MachineSummary])
def list_machines():
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")
    return ml_service.get_machine_list()

@app.get("/api/machines/{machine_id}")
def get_machine(machine_id: str):
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")
    machine = ml_service.get_machine_detail(machine_id)
    if not machine:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found.")
    return machine

@app.get("/api/machines/{machine_id}/sensors")
def get_machine_sensors(machine_id: str):
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")
    sensors = ml_service.get_machine_sensors(machine_id)
    if not sensors:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found.")
    return sensors

@app.get("/api/machines/{machine_id}/prediction")
def get_machine_prediction(machine_id: str):
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")
    pred = ml_service.get_machine_prediction(machine_id)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found.")
    return pred

@app.get("/api/machines/{machine_id}/anomalies")
def get_machine_anomalies(machine_id: str):
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")
    anom = ml_service.get_machine_anomalies(machine_id)
    if not anom:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found.")
    return anom

@app.get("/api/machines/{machine_id}/health")
def get_machine_health(machine_id: str):
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")
    h = ml_service.get_machine_health(machine_id)
    if not h:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found.")
    return h

@app.get("/api/analytics/models", response_model=AnalyticsModelsResponse)
def get_analytics_models():
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")
    return AnalyticsModelsResponse(
        dataset=ml_service.metadata.get("dataset", "C-MAPSS FD001"),
        metadata=ml_service.metadata,
        evaluation=ml_service.evaluation
    )

@app.post("/api/predict/rul", response_model=PredictRULResponse)
def predict_rul(req: PredictRULRequest):
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")

    if req.sequence is not None:
        if len(req.sequence) == 0:
            raise HTTPException(status_code=400, detail="Sequence matrix cannot be empty.")
        df = pd.DataFrame(req.sequence, columns=FEATURE_COLS[:len(req.sequence[0])])
    elif req.features is not None:
        if len(req.features) != len(FEATURE_COLS):
            raise HTTPException(status_code=400, detail=f"Expected {len(FEATURE_COLS)} features, got {len(req.features)}.")
        df = pd.DataFrame([req.features], columns=FEATURE_COLS)
    else:
        raise HTTPException(status_code=400, detail="Must provide 'features' or 'sequence'.")

    try:
        lstm_pred = ml_service.predict_rul_raw(df, use_lstm=True)
        rf_pred = ml_service.predict_rul_raw(df, use_lstm=False)
        mae = ml_service.evaluation.get("rul_models", {}).get("lstm", {}).get("mae", 11.3)
        return PredictRULResponse(
            predicted_rul=round(lstm_pred, 2),
            rf_predicted_rul=round(rf_pred, 2),
            model_used="PyTorch LSTM" if req.use_lstm and len(df) >= 30 else "Random Forest Baseline",
            confidence_interval={
                "min": max(0.0, round(lstm_pred - mae, 2)),
                "max": round(lstm_pred + mae, 2)
            }
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during RUL inference.")

@app.post("/api/detect/anomaly", response_model=DetectAnomalyResponse)
def detect_anomaly(req: DetectAnomalyRequest):
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")

    if len(req.features) != len(FEATURE_COLS):
        raise HTTPException(status_code=400, detail=f"Expected {len(FEATURE_COLS)} features, got {len(req.features)}.")

    df = pd.DataFrame([req.features], columns=FEATURE_COLS)
    try:
        anom_res = ml_service.detect_anomaly_raw(df)
        return DetectAnomalyResponse(
            anomaly_score=anom_res["anomaly_score"],
            severity=anom_res["severity"],
            status="active" if anom_res["is_anomaly"] else "normal",
            is_anomaly=anom_res["is_anomaly"]
        )
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during anomaly detection.")

# ---------------------------------------------------------
# Phase 5 Real Knowledge Base Endpoints
# ---------------------------------------------------------
@app.get("/api/knowledge/documents", response_model=List[KnowledgeDocumentResponse])
def list_knowledge_documents():
    docs = vector_store.get_documents()
    return docs

@app.get("/api/knowledge/documents/{doc_id}", response_model=KnowledgeDocumentResponse)
def get_knowledge_document(doc_id: str):
    doc = vector_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return doc

@app.get("/api/knowledge/documents/{doc_id}/chunks")
def get_document_chunks(doc_id: str):
    chunks = vector_store.get_chunks_for_document(doc_id)
    if not chunks and not vector_store.get_document(doc_id):
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return chunks

@app.post("/api/knowledge/upload", response_model=KnowledgeDocumentResponse)
async def upload_knowledge_document(
    file: UploadFile = File(...),
    category: Optional[str] = Form("Manual")
):
    try:
        contents = await file.read()
        raw_filename = file.filename or "uploaded_doc.txt"
        filename = sanitize_filename(raw_filename)
        
        # Validate and extract text
        pages, full_text = extract_text(filename, contents)
        
        # Generate document ID and metadata
        doc_id = f"DOC-{len(vector_store.documents) + 901:03d}"
        doc_name = os.path.splitext(filename)[0].replace('_', ' ').title()
        ext_type = os.path.splitext(filename)[1].replace('.', '').upper()

        # Generate semantic chunks with provenance
        chunks = chunk_pages(doc_id, doc_name, pages)

        doc_meta = {
            "id": doc_id,
            "title": doc_name,
            "type": ext_type,
            "size": len(contents),
            "uploadDate": pd.Timestamp.now().isoformat(),
            "processingStatus": "ready",
            "indexedChunks": len(chunks),
            "sourceCategory": category or "Manual"
        }

        # Add to vector store index
        vector_store.add_document(doc_meta, chunks)
        logger.info(f"Successfully uploaded and indexed document '{doc_name}' ({doc_id}) with {len(chunks)} chunks.")
        return doc_meta

    except DocumentExtractionError as de:
        raise HTTPException(status_code=400, detail=str(de))
    except Exception as e:
        logger.error(f"Document upload & processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Document upload and processing failed.")

@app.post("/api/knowledge/search", response_model=List[KnowledgeSearchResult])
def search_knowledge_base(req: KnowledgeSearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    
    results = vector_store.search(
        query=req.query,
        top_k=req.top_k or 5,
        min_score=req.min_score or 0.05
    )
    return results

# ---------------------------------------------------------
# Phase 5 Real AI Diagnostics Endpoints
# ---------------------------------------------------------
@app.post("/api/diagnostics", response_model=DiagnosticResponse)
def create_diagnostic_case(req: CreateDiagnosticRequest):
    if not ml_service.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not initialized.")

    machine_context = ml_service.get_machine_detail(req.machine_id)
    if not machine_context:
        raise HTTPException(status_code=404, detail=f"Machine '{req.machine_id}' not found.")

    # Run RAG diagnostic generation
    diag_res = rag_engine.generate_diagnostic(machine_context)
    
    # Store case in persistent history
    diagnostic_cases_cache[diag_res["id"]] = diag_res
    save_diagnostic_cases()
    logger.info(f"Generated diagnostic case '{diag_res['id']}' for {req.machine_id} using {diag_res.get('generatorUsed')}.")
    
    return diag_res

@app.get("/api/diagnostics", response_model=List[DiagnosticResponse])
def list_diagnostic_cases():
    return list(diagnostic_cases_cache.values())

@app.get("/api/diagnostics/{case_id}", response_model=DiagnosticResponse)
def get_diagnostic_case(case_id: str):
    case = diagnostic_cases_cache.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Diagnostic case '{case_id}' not found.")
    return case

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
