"""Evidence-Backed Maintenance Decision Support Service for MachinaSense Phase 7.

Combines verified ML predictions with grounded RAG technical document retrieval:
1. Observed Evidence: Real ML RUL, anomaly score, sensor readings, and fleet status
2. Supporting Documentation: Genuine citations with title, page, section, and excerpt
3. Suggested Diagnostic Checks: Actionable inspection procedures supported by evidence
4. Suggested Advisory Action: Maintenance recommendations with human authorization notices
5. Uncertainty & Limitations: Explicit missing evidence and statistical limitations
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models import (
    AnomalyEvent,
    DocumentChunk,
    Machine,
    MaintenanceTask,
    ModelPredictionRecord,
    Prediction,
    SensorReading,
    User,
)
from app.ml_service import ml_service, SENSOR_NAMES_MAP

logger = logging.getLogger("machinasense.decision_support")


def generate_decision_support(
    db: Session,
    user: User,
    machine_id: Optional[str] = None,
    prediction_id: Optional[str] = None,
    symptoms: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates an evidence-backed maintenance advisory report for an industrial machine or prediction."""
    machine: Optional[Machine] = None
    pred_record: Optional[ModelPredictionRecord] = None

    # 1. Resolve Machine Context if provided
    if machine_id:
        machine = db.query(Machine).filter(Machine.id == machine_id, Machine.user_id == user.id).first()
        if not machine:
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found.")

    # 2. Resolve Prediction Record if provided
    if prediction_id:
        pred_record = (
            db.query(ModelPredictionRecord)
            .filter(ModelPredictionRecord.id == prediction_id, ModelPredictionRecord.user_id == user.id)
            .first()
        )
        if not pred_record:
            raise HTTPException(status_code=404, detail=f"Prediction record '{prediction_id}' not found.")

    # Gather ML Observations & Sensor Facts
    rul_value: Optional[float] = None
    rul_model_name = "Unknown"
    confidence_interval: Optional[Dict[str, float]] = None
    anomaly_status = "none"
    anomaly_score = 0.0
    latest_reading: Optional[SensorReading] = None
    open_tasks: List[MaintenanceTask] = []

    if machine:
        latest_pred = (
            db.query(Prediction)
            .filter(Prediction.machine_id == machine.id, Prediction.user_id == user.id)
            .order_by(Prediction.created_at.desc())
            .first()
        )
        if latest_pred:
            rul_value = float(latest_pred.predicted_rul) if latest_pred.predicted_rul is not None else None
            rul_model_name = latest_pred.model_used or "NASA C-MAPSS FD001"
            if latest_pred.confidence_min is not None and latest_pred.confidence_max is not None:
                confidence_interval = {
                    "min_cycles": round(float(latest_pred.confidence_min), 1),
                    "max_cycles": round(float(latest_pred.confidence_max), 1),
                }

        latest_anom = (
            db.query(AnomalyEvent)
            .filter(AnomalyEvent.machine_id == machine.id, AnomalyEvent.user_id == user.id)
            .order_by(AnomalyEvent.timestamp.desc())
            .first()
        )
        if latest_anom:
            anomaly_status = latest_anom.severity
            anomaly_score = float(latest_anom.score)

        latest_reading = (
            db.query(SensorReading)
            .filter(SensorReading.machine_id == machine.id, SensorReading.user_id == user.id)
            .order_by(SensorReading.timestamp.desc())
            .first()
        )

        open_tasks = (
            db.query(MaintenanceTask)
            .filter(MaintenanceTask.machine_id == machine.id, MaintenanceTask.user_id == user.id)
            .all()
        )

    elif pred_record:
        pred_dict = json.loads(pred_record.prediction_json) if pred_record.prediction_json else {}
        if "rul_cycles" in pred_dict:
            rul_value = float(pred_dict["rul_cycles"])
            rul_model_name = pred_record.model.name if pred_record.model else "Active Model"
        if "is_anomaly" in pred_dict:
            anomaly_status = "critical" if pred_dict["is_anomaly"] else "healthy"
            anomaly_score = float(pred_dict.get("anomaly_score", 0.0))

    # Determine risk level based on verified RUL
    if rul_value is not None:
        if rul_value < 30.0:
            risk_level = "critical"
        elif rul_value < 70.0:
            risk_level = "warning"
        else:
            risk_level = "healthy"
    else:
        risk_level = "unknown"

    # Assemble Observed Evidence (Facts only)
    sensor_telemetry: Dict[str, Any] = {}
    if latest_reading:
        sensor_telemetry = {
            "vibration_s2": {
                "value": latest_reading.s2,
                "label": SENSOR_NAMES_MAP.get("s2", {}).get("name", "LPC Outlet Temp"),
                "unit": SENSOR_NAMES_MAP.get("s2", {}).get("unit", "°R"),
            },
            "temperature_s3": {
                "value": latest_reading.s3,
                "label": SENSOR_NAMES_MAP.get("s3", {}).get("name", "HPC Outlet Temp"),
                "unit": SENSOR_NAMES_MAP.get("s3", {}).get("unit", "°R"),
            },
            "pressure_s4": {
                "value": latest_reading.s4,
                "label": SENSOR_NAMES_MAP.get("s4", {}).get("name", "LPT Outlet Temp"),
                "unit": SENSOR_NAMES_MAP.get("s4", {}).get("unit", "°R"),
            },
            "pressure_s7": {
                "value": latest_reading.s7,
                "label": SENSOR_NAMES_MAP.get("s7", {}).get("name", "HPC Outlet Pressure"),
                "unit": SENSOR_NAMES_MAP.get("s7", {}).get("unit", "psia"),
            },
            "core_speed_s14": {
                "value": latest_reading.s14,
                "label": SENSOR_NAMES_MAP.get("s14", {}).get("name", "Corrected Core Speed"),
                "unit": SENSOR_NAMES_MAP.get("s14", {}).get("unit", "rpm"),
            },
            "cycle": latest_reading.cycle,
        }

    observed_evidence = {
        "machine_id": machine.id if machine else None,
        "machine_name": machine.name if machine else "Unspecified Equipment",
        "machine_type": machine.machine_type if machine else "Turbofan Subsystem",
        "predicted_rul_cycles": rul_value,
        "rul_model_used": rul_model_name,
        "confidence_interval": confidence_interval,
        "risk_level": risk_level,
        "anomaly_severity": anomaly_status,
        "anomaly_score": round(anomaly_score, 4),
        "telemetry_observations": sensor_telemetry,
        "open_maintenance_count": len(open_tasks),
    }

    # 3. Grounded Technical Document Retrieval (RAG)
    # Build query from symptoms and observed anomalies
    search_terms = []
    if symptoms:
        search_terms.append(symptoms)
    if risk_level in {"critical", "warning"}:
        search_terms.append(f"{risk_level} degradation turbofan stage")
    if anomaly_status in {"critical", "high"}:
        search_terms.append("bearing temperature pressure anomaly inspection")
    search_query = " ".join(search_terms) or "maintenance inspection procedures troubleshooting"

    retrieved_chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.user_id == user.id)
        .all()
    )

    supporting_docs: List[Dict[str, Any]] = []
    if retrieved_chunks:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            corpus = [c.content for c in retrieved_chunks]
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            X = vec.fit_transform(corpus)
            q_vec = vec.transform([search_query])
            sims = cosine_similarity(q_vec, X).flatten()

            for idx in sims.argsort()[::-1][:4]:
                score = float(sims[idx])
                if score >= 0.05:
                    c = retrieved_chunks[idx]
                    supporting_docs.append({
                        "chunk_id": c.id,
                        "document_id": c.document_id,
                        "document_title": c.document.title if c.document else "Technical Manual",
                        "section": c.section,
                        "page": c.page,
                        "relevance_score": round(score, 4),
                        "excerpt": c.content[:350].strip() + ("..." if len(c.content) > 350 else ""),
                    })
        except Exception as e:
            logger.warning("TF-IDF retrieval error in decision support: %s", e)

    # 4. Synthesize Suggested Checks (Grounded in evidence)
    suggested_checks: List[Dict[str, Any]] = []
    if risk_level == "critical" or (rul_value is not None and rul_value < 30.0):
        suggested_checks.append({
            "step": 1,
            "category": "Immediate Visual & Borescope Inspection",
            "procedure": "Conduct video-borescope inspection of high-pressure compressor (HPC) stage blades and stator vanes.",
            "rationale": f"Estimated RUL has fallen to {rul_value:.1f} cycles (critical threshold < 30 cycles).",
            "priority": "High",
        })
        suggested_checks.append({
            "step": 2,
            "category": "Sensor Signal Validation",
            "procedure": "Verify calibration and wiring harness continuity for thermocouple channels (s2, s3) and pressure transducers (s4, s7).",
            "rationale": "Confirms observed telemetry shifts reflect real physical thermal degradation rather than instrumentation drift.",
            "priority": "High",
        })
    elif risk_level == "warning":
        suggested_checks.append({
            "step": 1,
            "category": "Trend Analysis & Lubrication Check",
            "procedure": "Sample and analyze lubrication oil for particulate contamination and metallic debris.",
            "rationale": f"Predicted RUL is in warning zone ({rul_value:.1f} cycles).",
            "priority": "Medium",
        })
    else:
        suggested_checks.append({
            "step": 1,
            "category": "Routine Telemetry Monitoring",
            "procedure": "Continue standard operational logging and verify automated anomaly detection triggers.",
            "rationale": "Equipment telemetry remains within healthy statistical bounds.",
            "priority": "Routine",
        })

    # If anomaly detected, add targeted anomaly check
    if anomaly_status in {"critical", "high"}:
        suggested_checks.append({
            "step": len(suggested_checks) + 1,
            "category": "Anomaly Isolation",
            "procedure": "Inspect mechanical seals and bleed air enthalpy valves (s17) for pressure bypass leakage.",
            "rationale": f"Isolation Forest identified significant pattern anomaly (Severity: {anomaly_status}, Score: {anomaly_score:.2f}).",
            "priority": "High",
        })

    # 5. Suggested Advisory Action
    if risk_level == "critical":
        suggested_action = {
            "action_type": "Immediate Scheduled Overhaul",
            "recommended_timeframe": "Within next 5-10 operational cycles",
            "summary": "Schedule machinery downtime for component overhaul and seal replacement before complete fatigue failure.",
            "requires_human_approval": True,
            "advisory_disclaimer": "Advisory recommendation only. Qualified engineering personnel must approve maintenance tickets.",
        }
    elif risk_level == "warning":
        suggested_action = {
            "action_type": "Planned Preventive Maintenance",
            "recommended_timeframe": "Schedule during next planned maintenance window (within 20-30 cycles)",
            "summary": "Order replacement wear components and arrange for intermediate inspection during scheduled downtime.",
            "requires_human_approval": True,
            "advisory_disclaimer": "Advisory recommendation only. Qualified engineering personnel must approve maintenance tickets.",
        }
    else:
        suggested_action = {
            "action_type": "Continue Normal Operations",
            "recommended_timeframe": "Next routine maintenance interval",
            "summary": "No immediate mechanical intervention required. Maintain real-time telemetry observation.",
            "requires_human_approval": False,
            "advisory_disclaimer": "Advisory recommendation only.",
        }

    # 6. Uncertainty & Limitations
    uncertainties = [
        "RUL estimates are statistical projections derived from mathematical model patterns; they are not guaranteed failure timestamps.",
        "Unsupervised anomaly detection highlights telemetry pattern divergence and does not indicate confirmed mechanical breakage.",
    ]
    if not supporting_docs:
        uncertainties.append(
            "No matching technical manuals or engineering SOPs were found in the private knowledge base. "
            "Upload technical documentation in the Knowledge Base to enrich grounded maintenance citations."
        )
    if not latest_reading:
        uncertainties.append("Recent sensor readings were unavailable; analysis relied on static machine fleet history.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observed_evidence": observed_evidence,
        "supporting_documentation": supporting_docs,
        "suggested_checks": suggested_checks,
        "suggested_action": suggested_action,
        "uncertainties_and_limitations": uncertainties,
        "compliance_notice": (
            "MachinaSense Decision Support is an advisory engineering tool. All recommendations require "
            "review and authorization by certified maintenance technicians before executing physical interventions."
        ),
    }
