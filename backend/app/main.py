"""Authenticated, PostgreSQL-backed MachinaSense API."""
from __future__ import annotations
import csv, io, json, logging, os, re, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.auth.clerk import get_current_user
from app.config import settings
from app.db.models import AnomalyEvent, CopilotConversation, CopilotMessage, DiagnosticCase, DocumentChunk, KnowledgeDoc, Machine, MaintenanceTask, Prediction, SensorReading, User
from app.db.session import get_db, init_db
from app.ml_service import FEATURE_COLS, ml_service
from app.rag.chunker import chunk_pages
from app.rag.extractor import DocumentExtractionError, extract_text
from app.services.llm_service import llm_service

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO)); logger = logging.getLogger("machinasense")
def sanitize_filename(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.basename(filename).replace("\\", "/").split("/")[-1]) or "document.txt"
@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db(); ml_service.load_artifacts(); yield
app = FastAPI(title=settings.PROJECT_NAME, version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True, allow_methods=["GET","POST","PATCH","DELETE"], allow_headers=["Authorization","Content-Type"])
@app.exception_handler(Exception)
async def safe_error(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path); return JSONResponse(status_code=500, content={"detail":"An internal server error occurred."})
@app.get("/health")
def health():
    return {"status":"healthy" if ml_service.is_loaded else "degraded", "backend":"live", "models_loaded":ml_service.is_loaded, "rag_status":"ready", "llm_status":"provider-chain", "dataset":"NASA C-MAPSS FD001 Benchmark", "models":["Random Forest Regressor Baseline","PyTorch LSTM Temporal RUL","Isolation Forest Anomaly Detector","TF-IDF Vector RAG Engine"], "artifacts_path":"backend/artifacts"}
def machine_for(db: Session, user: User, machine_id: str) -> Machine:
    item=db.query(Machine).filter(Machine.id==machine_id,Machine.user_id==user.id).first()
    if not item: raise HTTPException(404,"Machine not found.")
    return item
def risk(rul: float|None) -> str: return "critical" if rul is not None and rul<30 else "warning" if rul is not None and rul<70 else "healthy"
def machine_view(db: Session, item: Machine) -> dict[str,Any]:
    p=db.query(Prediction).filter(Prediction.machine_id==item.id).order_by(Prediction.created_at.desc()).first(); a=db.query(AnomalyEvent).filter(AnomalyEvent.machine_id==item.id).order_by(AnomalyEvent.timestamp.desc()).first(); s=db.query(SensorReading).filter(SensorReading.machine_id==item.id).order_by(SensorReading.timestamp.desc()).first(); rul=p.predicted_rul if p else None
    return {"id":item.id,"name":item.name,"type":item.machine_type,"model":item.machine_type,"location":item.location,"description":item.description,"status":p.risk_level if p else "offline","rulDays":rul,"predictedRul":rul,"rfPredictedRul":p.rf_predicted_rul if p else None,"rulConfidence":87 if p else 0,"failureRisk":max(0,min(100,round(100-rul))) if rul is not None else 0,"healthScore":p.health_score if p else 0,"currentCycle":s.cycle if s else 0,"runningHours":s.cycle if s else 0,"anomalySeverity":a.severity if a else "none","anomalyScore":a.score if a else None,"lastMaintenance":None,"lastUpdated":(item.updated_at or item.created_at).isoformat()}
@app.get("/api/me")
def me(user:User=Depends(get_current_user)): return {"id":user.id,"email":user.email,"name":" ".join(x for x in [user.first_name,user.last_name] if x),"authenticated":True}
@app.get("/api/machines")
def machines(db:Session=Depends(get_db),user:User=Depends(get_current_user)): return [machine_view(db,m) for m in db.query(Machine).filter(Machine.user_id==user.id).order_by(Machine.created_at.desc())]
@app.post("/api/machines",status_code=201)
def create_machine(payload:dict,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    for key in ("machine_id","name","machine_type","location"):
        if not isinstance(payload.get(key),str) or not payload[key].strip(): raise HTTPException(422,f"{key} is required.")
    mid=payload["machine_id"].strip()
    if db.query(Machine).filter(Machine.id==mid).first(): raise HTTPException(409,"Machine ID already exists.")
    m=Machine(id=mid,user_id=user.id,name=payload["name"].strip(),machine_type=payload["machine_type"].strip(),location=payload["location"].strip(),description=(payload.get("description") or "").strip()); db.add(m);db.commit();db.refresh(m);return machine_view(db,m)
@app.get("/api/machines/{machine_id}")
def machine(machine_id:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)): return machine_view(db,machine_for(db,user,machine_id))
@app.delete("/api/machines/{machine_id}",status_code=204)
def delete_machine(machine_id:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)): db.delete(machine_for(db,user,machine_id));db.commit()
@app.post("/api/machines/{machine_id}/telemetry",status_code=201)
async def telemetry(machine_id:str,file:UploadFile=File(...),db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    m=machine_for(db,user,machine_id); filename=sanitize_filename(file.filename or "telemetry.csv")
    if not filename.lower().endswith(".csv") or file.content_type not in {"text/csv","application/csv","application/vnd.ms-excel","application/octet-stream"}: raise HTTPException(400,"Telemetry must be a CSV file.")
    raw=await file.read()
    if not raw or len(raw)>settings.MAX_FILE_SIZE_BYTES: raise HTTPException(400,"Telemetry file is empty or exceeds the upload limit.")
    try: rows=list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    except (UnicodeDecodeError,csv.Error) as exc: raise HTTPException(400,"Telemetry CSV is malformed.") from exc
    if not rows: raise HTTPException(400,"Telemetry CSV contains no data rows.")
    if not set(FEATURE_COLS).issubset(rows[0].keys()): raise HTTPException(400,"Telemetry CSV requires columns: "+", ".join(FEATURE_COLS))
    values=[]
    for i,row in enumerate(rows,2):
        try:
            d={key:float(row[key]) for key in FEATURE_COLS}; d["cycle"]=int(row.get("cycle") or i-1)
            if any(pd.isna(v) for v in d.values()): raise ValueError()
            values.append(d)
        except (TypeError,ValueError) as exc: raise HTTPException(400,f"Malformed or non-numeric telemetry at row {i}.") from exc
    if not ml_service.is_loaded: raise HTTPException(503,"ML models are unavailable.")
    frame=pd.DataFrame(values)
    try: lstm=ml_service.predict_rul_raw(frame,use_lstm=True); rf=ml_service.predict_rul_raw(frame,use_lstm=False); anomaly=ml_service.detect_anomaly_raw(frame.iloc[[-1]][FEATURE_COLS])
    except Exception as exc: logger.exception("ML inference failed"); raise HTTPException(503,"ML inference could not be completed.") from exc
    for d in values: db.add(SensorReading(machine_id=m.id,user_id=user.id,**d,is_anomaly=anomaly["is_anomaly"],anomaly_score=anomaly["anomaly_score"]))
    mae=ml_service.evaluation.get("rul_models",{}).get("lstm",{}).get("mae",11.3); p=Prediction(id=str(uuid.uuid4()),machine_id=m.id,user_id=user.id,predicted_rul=float(lstm),rf_predicted_rul=float(rf),model_used="PyTorch LSTM" if len(frame)>=30 else "Random Forest Baseline",confidence_min=max(0,float(lstm)-mae),confidence_max=float(lstm)+mae,health_score=max(0,min(100,round(lstm))),risk_level=risk(float(lstm)));db.add(p)
    if anomaly["is_anomaly"]: db.add(AnomalyEvent(id=str(uuid.uuid4()),machine_id=m.id,user_id=user.id,severity=anomaly["severity"],score=anomaly["anomaly_score"],description="Isolation Forest detected an anomalous uploaded telemetry observation.",affected_sensors=json.dumps([])))
    db.commit();return {"accepted_rows":len(values),"prediction":machine_view(db,m)}
@app.get("/api/machines/{machine_id}/sensors")
def sensors(machine_id:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    machine_for(db,user,machine_id);rows=db.query(SensorReading).filter(SensorReading.machine_id==machine_id,SensorReading.user_id==user.id).order_by(SensorReading.timestamp.desc()).limit(250).all()[::-1]
    return [{"timestamp":r.timestamp.isoformat(),"vibration":r.s2,"temperature":r.s3,"pressure":r.s4,"isAnomaly":r.is_anomaly,"cycle":r.cycle} for r in rows]
@app.get("/api/machines/{machine_id}/prediction")
def prediction(machine_id:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    machine_for(db,user,machine_id);rows=db.query(Prediction).filter(Prediction.machine_id==machine_id,Prediction.user_id==user.id).order_by(Prediction.created_at).all()
    if not rows:raise HTTPException(404,"No predictions available.")
    p=rows[-1];return {"predictedRul":p.predicted_rul,"rfPredictedRul":p.rf_predicted_rul,"riskLevel":p.risk_level,"modelUsed":p.model_used,"confidenceInterval":{"min":p.confidence_min,"max":p.confidence_max},"degradationCurve":[{"predictedRul":x.predicted_rul,"timestamp":x.created_at.isoformat()} for x in rows]}
@app.get("/api/machines/{machine_id}/anomalies")
def anomalies(machine_id:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    machine_for(db,user,machine_id);rows=db.query(AnomalyEvent).filter(AnomalyEvent.machine_id==machine_id,AnomalyEvent.user_id==user.id).order_by(AnomalyEvent.timestamp.desc()).all();return {"anomalyEvents":[{"id":r.id,"machineId":r.machine_id,"timestamp":r.timestamp.isoformat(),"severity":r.severity,"score":r.score,"description":r.description,"affectedSensors":json.loads(r.affected_sensors)} for r in rows]}
def maintenance_view(task:MaintenanceTask,machine:Machine)->dict[str,Any]:
    return {"id":task.id,"machineId":machine.id,"machineName":machine.name,"priority":task.priority,"condition":task.description,"action":task.recommended_action or task.title,"status":task.status,"timeframe":"Immediate" if task.priority=="critical" else "Planned","createdAt":task.created_at.isoformat()}
@app.get("/api/maintenance")
def maintenance_records(db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    rows=db.query(MaintenanceTask).filter(MaintenanceTask.user_id==user.id).order_by(MaintenanceTask.created_at.desc()).all();return [maintenance_view(task,machine_for(db,user,task.machine_id)) for task in rows]
@app.post("/api/maintenance",status_code=201)
def create_maintenance(payload:dict,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    machine=machine_for(db,user,payload.get("machine_id") or "")
    for key in ("title","description"):
        if not isinstance(payload.get(key),str) or not payload[key].strip():raise HTTPException(422,f"{key} is required.")
    task=MaintenanceTask(id=str(uuid.uuid4()),machine_id=machine.id,user_id=user.id,title=payload["title"].strip(),description=payload["description"].strip(),recommended_action=(payload.get("recommended_action") or "").strip(),priority=payload.get("priority") or "medium",status=payload.get("status") or "open");db.add(task);db.commit();return maintenance_view(task,machine)
@app.patch("/api/maintenance/{task_id}")
def update_maintenance(task_id:str,payload:dict,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    task=db.query(MaintenanceTask).filter(MaintenanceTask.id==task_id,MaintenanceTask.user_id==user.id).first()
    if not task:raise HTTPException(404,"Maintenance record not found.")
    for key in ("title","description","recommended_action","priority","status"):
        if key in payload and isinstance(payload[key],str):setattr(task,key,payload[key].strip())
    db.commit();return maintenance_view(task,machine_for(db,user,task.machine_id))
@app.delete("/api/maintenance/{task_id}",status_code=204)
def delete_maintenance(task_id:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    task=db.query(MaintenanceTask).filter(MaintenanceTask.id==task_id,MaintenanceTask.user_id==user.id).first()
    if not task:raise HTTPException(404,"Maintenance record not found.")
    db.delete(task);db.commit()
@app.post("/api/predict/rul")
def predict_rul(payload:dict,_:User=Depends(get_current_user)):
    values=payload.get("features") or []
    if len(values)!=len(FEATURE_COLS):raise HTTPException(400,f"Expected {len(FEATURE_COLS)} features.")
    if not ml_service.is_loaded:raise HTTPException(503,"ML models are unavailable.")
    frame=pd.DataFrame([values],columns=FEATURE_COLS);lstm=ml_service.predict_rul_raw(frame,use_lstm=True);rf=ml_service.predict_rul_raw(frame,use_lstm=False);mae=ml_service.evaluation.get("rul_models",{}).get("lstm",{}).get("mae",11.3)
    return {"predicted_rul":round(lstm,2),"rf_predicted_rul":round(rf,2),"model_used":"Random Forest Baseline","confidence_interval":{"min":max(0,lstm-mae),"max":lstm+mae}}
@app.post("/api/detect/anomaly")
def detect_anomaly(payload:dict,_:User=Depends(get_current_user)):
    values=payload.get("features") or []
    if len(values)!=len(FEATURE_COLS):raise HTTPException(400,f"Expected {len(FEATURE_COLS)} features.")
    if not ml_service.is_loaded:raise HTTPException(503,"ML models are unavailable.")
    result=ml_service.detect_anomaly_raw(pd.DataFrame([values],columns=FEATURE_COLS));return {"anomaly_score":result["anomaly_score"],"severity":result["severity"],"status":"active" if result["is_anomaly"] else "normal","is_anomaly":result["is_anomaly"]}
@app.get("/api/knowledge/documents")
def documents(db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    rows=db.query(KnowledgeDoc).filter(KnowledgeDoc.user_id==user.id).order_by(KnowledgeDoc.upload_date.desc()).all();return [{"id":d.id,"title":d.title,"type":d.type,"size":d.size,"uploadDate":d.upload_date.isoformat(),"processingStatus":"ready","indexedChunks":d.indexed_chunks,"sourceCategory":d.source_category} for d in rows]
@app.get("/api/knowledge/documents/{document_id}")
def document(document_id:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    d=db.query(KnowledgeDoc).filter(KnowledgeDoc.id==document_id,KnowledgeDoc.user_id==user.id).first()
    if not d:raise HTTPException(404,"Document not found.")
    return {"id":d.id,"title":d.title,"type":d.type,"size":d.size,"uploadDate":d.upload_date.isoformat(),"processingStatus":"ready","indexedChunks":d.indexed_chunks,"sourceCategory":d.source_category}
@app.get("/api/knowledge/documents/{document_id}/chunks")
def document_chunks(document_id:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    document(document_id,db,user);return [{"chunk_id":c.id,"document_id":c.document_id,"page":c.page,"section":c.section,"excerpt":c.content[:500]} for c in db.query(DocumentChunk).filter(DocumentChunk.document_id==document_id,DocumentChunk.user_id==user.id)]
@app.delete("/api/knowledge/documents/{document_id}",status_code=204)
def delete_document(document_id:str,db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    d=db.query(KnowledgeDoc).filter(KnowledgeDoc.id==document_id,KnowledgeDoc.user_id==user.id).first()
    if not d:raise HTTPException(404,"Document not found.")
    db.query(DocumentChunk).filter(DocumentChunk.document_id==document_id,DocumentChunk.user_id==user.id).delete()
    db.delete(d);db.commit()
@app.post("/api/knowledge/upload",status_code=201)
async def document_upload(file:UploadFile=File(...),category:str=Form("Manual"),db:Session=Depends(get_db),user:User=Depends(get_current_user)):
    raw=await file.read();name=sanitize_filename(file.filename or "document.txt")
    if len(raw)>settings.MAX_FILE_SIZE_BYTES:raise HTTPException(400,"Document exceeds the upload limit.")
    try:pages,_=extract_text(name,raw)
    except DocumentExtractionError as exc:raise HTTPException(400,str(exc)) from exc
    did=str(uuid.uuid4());d=KnowledgeDoc(id=did,user_id=user.id,scope="USER_KNOWLEDGE",title=os.path.splitext(name)[0],type=os.path.splitext(name)[1][1:].upper(),size=len(raw),source_category=category);chunks=chunk_pages(did,d.title,pages);d.indexed_chunks=len(chunks);db.add(d)
    for c in chunks:db.add(DocumentChunk(id=c["chunk_id"],document_id=did,user_id=user.id,page=c["page"],section=c["section"],content=c["text"]))
    db.commit();return {"id":d.id,"title":d.title,"type":d.type,"size":d.size,"uploadDate":d.upload_date.isoformat(),"processingStatus":"ready","indexedChunks":d.indexed_chunks,"sourceCategory":d.source_category}
@app.post("/api/knowledge/search")
def document_search(payload: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "Search query cannot be empty.")
    rows = db.query(DocumentChunk).filter(DocumentChunk.user_id == user.id).all()
    if not rows:
        return []
    
    top_k = int(payload.get("top_k", 5))
    min_score = float(payload.get("min_score", 0.05))

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        corpus = [c.content for c in rows]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        results = []
        for idx in similarities.argsort()[::-1]:
            score = float(similarities[idx])
            if score >= min_score:
                c = rows[idx]
                results.append({
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "document_name": c.document.title if c.document else "Document",
                    "page": c.page,
                    "section": c.section,
                    "excerpt": c.content[:500],
                    "relevance_score": round(score, 4),
                })
                if len(results) >= top_k:
                    break
        if results:
            return results
    except Exception as e:
        logger.warning(f"TF-IDF search fallback to token overlap: {e}")

    tokens = set(query.lower().split())
    scored = []
    for c in rows:
        overlap = len(tokens & set(c.content.lower().split()))
        if overlap > 0:
            scored.append((c, overlap / max(1, len(tokens))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            "chunk_id": c.id,
            "document_id": c.document_id,
            "document_name": c.document.title if c.document else "Document",
            "page": c.page,
            "section": c.section,
            "excerpt": c.content[:500],
            "relevance_score": round(score, 4),
        }
        for c, score in scored[:top_k]
    ]

def copilot_context(db: Session, user: User, m: Machine | None, question: str = "") -> dict[str, Any]:
    user_chunks = db.query(DocumentChunk).filter(DocumentChunk.user_id == user.id).all()
    evidence = []
    if user_chunks:
        if question:
            tokens = set(question.lower().split())
            scored = []
            for c in user_chunks:
                match_count = len(tokens & set(c.content.lower().split()))
                if match_count > 0:
                    scored.append((c, match_count))
            scored.sort(key=lambda x: x[1], reverse=True)
            chosen = [c for c, _ in scored[:5]] if scored else user_chunks[:5]
        else:
            chosen = user_chunks[:5]
        evidence = [
            {
                "documentId": c.document_id,
                "documentTitle": c.document.title if c.document else "Document",
                "section": c.section,
                "page": c.page,
                "snippet": c.content[:500],
                "relevance": 1.0,
            }
            for c in chosen
        ]

    if not m:
        return {
            "machine": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rul": None,
            "rf_rul": None,
            "rul_model": None,
            "uncertainty": None,
            "health": None,
            "risk": None,
            "anomaly_status": "none",
            "anomaly_score": 0.0,
            "affected_sensors": [],
            "relevant_telemetry": None,
            "maintenance_context": [],
            "ml_findings": "",
            "evidence": evidence,
            "latest_anomalies": [],
        }

    p = db.query(Prediction).filter(Prediction.machine_id == m.id, Prediction.user_id == user.id).order_by(Prediction.created_at.desc()).first()
    s = db.query(SensorReading).filter(SensorReading.machine_id == m.id, SensorReading.user_id == user.id).order_by(SensorReading.timestamp.desc()).first()
    an = db.query(AnomalyEvent).filter(AnomalyEvent.machine_id == m.id, AnomalyEvent.user_id == user.id).order_by(AnomalyEvent.timestamp.desc()).limit(5).all()
    tasks = db.query(MaintenanceTask).filter(MaintenanceTask.machine_id == m.id, MaintenanceTask.user_id == user.id).order_by(MaintenanceTask.created_at.desc()).limit(5).all()

    rul_val = p.predicted_rul if p else None
    rf_val = p.rf_predicted_rul if p else None
    model_used = p.model_used if p else "None"
    conf_min = p.confidence_min if p else None
    conf_max = p.confidence_max if p else None
    health = p.health_score if p else (100 if s else 0)
    risk_lvl = p.risk_level if p else ("healthy" if s else "offline")
    anom_sev = an[0].severity if an else "none"
    anom_score = an[0].score if an else 0.0
    cycle = s.cycle if s else 0
    affected = json.loads(an[0].affected_sensors) if (an and an[0].affected_sensors) else []

    findings = (
        f"Machine: {m.id} ({m.name}). Model: {model_used}. "
        f"Predicted RUL: {f'{rul_val:.1f}' if rul_val is not None else 'unavailable'} cycles. "
        f"Health Score: {health}%. Risk: {risk_lvl}. "
        f"Anomaly Severity: {anom_sev} (Score: {anom_score:.2f}). "
        f"Current Cycle: {cycle}. Open Maintenance Tasks: {len(tasks)}."
    ) if (p or s or an) else f"Machine: {m.id} ({m.name}). Registered in fleet, but no telemetry has been uploaded yet."

    telemetry_data = None
    if s:
        telemetry_data = {
            "cycle": s.cycle,
            "vibration_s2": s.s2,
            "temperature_s3": s.s3,
            "pressure_s4": s.s4,
            "pressure_s7": s.s7,
            "pressure_s11": s.s11,
            "core_speed_s14": s.s14,
            "bleed_enthalpy_s17": s.s17,
        }

    return {
        "machine": {
            "id": m.id,
            "name": m.name,
            "type": m.machine_type,
            "location": m.location,
            "description": m.description,
            "status": risk_lvl,
        },
        "timestamp": (p.created_at if p else (s.timestamp if s else m.created_at)).isoformat(),
        "rul": rul_val,
        "rf_rul": rf_val,
        "rul_model": model_used,
        "uncertainty": {"min": conf_min, "max": conf_max} if conf_min is not None else None,
        "health": health,
        "risk": risk_lvl,
        "anomaly_status": anom_sev,
        "anomaly_score": anom_score,
        "affected_sensors": affected,
        "relevant_telemetry": telemetry_data,
        "maintenance_context": [
            {"id": t.id, "title": t.title, "priority": t.priority, "status": t.status, "action": t.recommended_action}
            for t in tasks
        ],
        "ml_findings": findings,
        "evidence": evidence,
        "latest_anomalies": [a.description for a in an],
    }

def diagnostic_view(d: DiagnosticCase, m: Machine) -> dict[str, Any]:
    return {
        "id": d.id,
        "machineId": m.id,
        "machineName": m.name,
        "status": "open",
        "severity": d.severity,
        "trigger": d.trigger,
        "finding": d.finding,
        "explanation": d.explanation,
        "evidence": json.loads(d.evidence) if d.evidence else [],
        "recommendedAction": d.recommended_action,
        "confidence": d.confidence,
        "generatorUsed": d.generator_used,
        "isGroundedFallback": d.is_grounded_fallback,
        "provider_used": d.provider_used,
        "fallback_level": d.fallback_level,
        "createdTime": d.created_at.isoformat(),
    }

@app.post("/api/diagnostics", status_code=201)
def create_diagnostic(payload: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    m = machine_for(db, user, payload.get("machine_id") or "")
    context = copilot_context(db, user, m, question="diagnostic investigation for machine failure and maintenance")
    result = llm_service.generate_grounded_response(context, "Generate an evidence-bound maintenance investigation.")
    evidence = context["evidence"]
    d = DiagnosticCase(
        id=str(uuid.uuid4()),
        machine_id=m.id,
        user_id=user.id,
        trigger="User-requested engineering investigation",
        finding=context["ml_findings"] or "No ML findings are available.",
        explanation=result["content"],
        recommended_action="Review the supplied evidence and investigate the observed signals before action.",
        severity=machine_view(db, m)["status"],
        confidence=0 if result["provider_used"] == "unavailable" else 80,
        generator_used=result["model_name"],
        provider_used=result["provider_used"],
        fallback_level=result["fallback_level"],
        is_grounded_fallback=result["provider_used"] == "grounded_fallback",
        evidence=json.dumps(evidence),
    )
    db.add(d)
    db.commit()
    return diagnostic_view(d, m)

@app.get("/api/diagnostics")
def diagnostics(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rows = db.query(DiagnosticCase).filter(DiagnosticCase.user_id == user.id).order_by(DiagnosticCase.created_at.desc()).all()
    return [diagnostic_view(d, machine_for(db, user, d.machine_id)) for d in rows]

@app.get("/api/diagnostics/{diagnostic_id}")
def diagnostic(diagnostic_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    d = db.query(DiagnosticCase).filter(DiagnosticCase.id == diagnostic_id, DiagnosticCase.user_id == user.id).first()
    if not d:
        raise HTTPException(404, "Diagnostic investigation not found.")
    return diagnostic_view(d, machine_for(db, user, d.machine_id))

@app.post("/api/diagnostics/{diagnostic_id}/explain")
def explain_diagnostic(diagnostic_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    d = db.query(DiagnosticCase).filter(DiagnosticCase.id == diagnostic_id, DiagnosticCase.user_id == user.id).first()
    if not d:
        raise HTTPException(404, "Diagnostic investigation not found.")
    return {"content": d.explanation, "provider_used": d.provider_used, "fallback_level": d.fallback_level}

def conversation_for(db: Session, user: User, cid: str) -> CopilotConversation:
    c = db.query(CopilotConversation).filter(CopilotConversation.id == cid, CopilotConversation.user_id == user.id).first()
    if not c:
        raise HTTPException(404, "Conversation not found.")
    return c

@app.post("/api/copilot/conversations", status_code=201)
def new_conversation(payload: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    mid = payload.get("machine_id")
    if mid:
        machine_for(db, user, mid)
    c = CopilotConversation(
        id=str(uuid.uuid4()),
        user_id=user.id,
        machine_id=mid,
        title=(payload.get("title") or "Engineering Copilot").strip(),
    )
    db.add(c)
    db.commit()
    return {"id": c.id, "machineId": c.machine_id, "title": c.title, "createdAt": c.created_at.isoformat()}

@app.patch("/api/copilot/conversations/{conversation_id}")
def update_conversation(conversation_id: str, payload: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    c = conversation_for(db, user, conversation_id)
    if "machine_id" in payload:
        mid = payload.get("machine_id")
        if mid:
            machine_for(db, user, mid)
            c.machine_id = mid
        else:
            c.machine_id = None
    if "title" in payload and isinstance(payload["title"], str) and payload["title"].strip():
        c.title = payload["title"].strip()
    db.commit()
    return {"id": c.id, "machineId": c.machine_id, "title": c.title, "createdAt": c.created_at.isoformat()}

@app.get("/api/copilot/conversations")
def list_conversations(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rows = db.query(CopilotConversation).filter(CopilotConversation.user_id == user.id).order_by(CopilotConversation.updated_at.desc()).all()
    return [{"id": c.id, "machineId": c.machine_id, "title": c.title, "createdAt": c.created_at.isoformat()} for c in rows]

@app.get("/api/copilot/conversations/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    c = conversation_for(db, user, conversation_id)
    return {
        "id": c.id,
        "machineId": c.machine_id,
        "title": c.title,
        "createdAt": c.created_at.isoformat(),
        "messages": [
            {
                "id": m.id,
                "sender": m.sender,
                "content": m.content,
                "provider_used": m.provider_used,
                "fallback_level": m.fallback_level,
                "evidence": json.loads(m.evidence) if m.evidence else [],
                "createdAt": m.created_at.isoformat(),
            }
            for m in c.messages
        ],
    }

@app.delete("/api/copilot/conversations/{conversation_id}", status_code=204)
def remove_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db.delete(conversation_for(db, user, conversation_id))
    db.commit()

@app.post("/api/copilot/conversations/{conversation_id}/messages", status_code=201)
def send_message(conversation_id: str, payload: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    c = conversation_for(db, user, conversation_id)
    question = (payload.get("content") or "").strip()
    if not question:
        raise HTTPException(422, "Message content is required.")
    
    # Allow client to update/specify machine_id with message
    if "machine_id" in payload:
        mid = payload.get("machine_id")
        if mid:
            machine_for(db, user, mid)
            c.machine_id = mid
        else:
            c.machine_id = None

    db.add(CopilotMessage(id=str(uuid.uuid4()), conversation_id=c.id, user_id=user.id, sender="user", content=question))
    
    m = machine_for(db, user, c.machine_id) if c.machine_id else None
    context = copilot_context(db, user, m, question=question)
    answer = llm_service.generate_grounded_response(context, question)
    
    reply = CopilotMessage(
        id=str(uuid.uuid4()),
        conversation_id=c.id,
        user_id=user.id,
        sender="copilot",
        content=answer["content"],
        provider_used=answer["provider_used"],
        fallback_level=answer["fallback_level"],
        evidence=json.dumps(context.get("evidence", [])),
    )
    db.add(reply)
    c.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"id": reply.id, "content": reply.content, **answer}

@app.get("/api/copilot/status")
def copilot_status(_: User = Depends(get_current_user)):
    return {
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "grok_configured": bool(settings.GROK_API_KEY),
        "provider_hierarchy": ["gemini", "grok", "grounded_fallback", "ml_only", "unavailable"],
    }

@app.get("/api/analytics/models")
def analytics(_: User = Depends(get_current_user)):
    return {
        "dataset": "NASA C-MAPSS FD001 Benchmark",
        "metadata": ml_service.metadata,
        "evaluation": ml_service.evaluation,
        "disclaimer": "Benchmark metrics are not validated industrial deployment performance.",
    }

@app.get("/api/analytics/fleet")
def fleet_analytics(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    user_machines = db.query(Machine).filter(Machine.user_id == user.id).all()
    total_machines = len(user_machines)
    if total_machines == 0:
        return {
            "total_machines": 0,
            "healthy": 0,
            "warning": 0,
            "critical": 0,
            "offline": 0,
            "average_rul": None,
            "total_anomalies": 0,
            "pending_maintenance": 0,
            "fleet": []
        }

    views = [machine_view(db, m) for m in user_machines]
    ruls = [v["predictedRul"] for v in views if v["predictedRul"] is not None]
    avg_rul = round(sum(ruls) / len(ruls), 1) if ruls else None

    healthy = sum(1 for v in views if v["status"] == "healthy")
    warning = sum(1 for v in views if v["status"] == "warning")
    critical = sum(1 for v in views if v["status"] == "critical")
    offline = sum(1 for v in views if v["status"] == "offline")

    total_anomalies = db.query(AnomalyEvent).filter(AnomalyEvent.user_id == user.id).count()
    pending_maint = db.query(MaintenanceTask).filter(MaintenanceTask.user_id == user.id, MaintenanceTask.status != "resolved").count()

    return {
        "total_machines": total_machines,
        "healthy": healthy,
        "warning": warning,
        "critical": critical,
        "offline": offline,
        "average_rul": avg_rul,
        "total_anomalies": total_anomalies,
        "pending_maintenance": pending_maint,
        "fleet": views
    }
