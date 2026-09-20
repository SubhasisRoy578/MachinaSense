import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.db.session import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)  # Clerk user ID, e.g. user_2...
    email = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    machines = relationship("Machine", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("CopilotConversation", back_populates="user", cascade="all, delete-orphan")

class Machine(Base):
    __tablename__ = "machines"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    name = Column(String, nullable=False)
    machine_type = Column(String, default="Turbofan Engine")
    location = Column(String, default="Industrial Plant")
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="machines")
    sensor_readings = relationship("SensorReading", back_populates="machine", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="machine", cascade="all, delete-orphan")
    anomalies = relationship("AnomalyEvent", back_populates="machine", cascade="all, delete-orphan")
    maintenance_tasks = relationship("MaintenanceTask", back_populates="machine", cascade="all, delete-orphan")
    diagnostics = relationship("DiagnosticCase", back_populates="machine", cascade="all, delete-orphan")
    conversations = relationship("CopilotConversation", back_populates="machine")

class SensorReading(Base):
    __tablename__ = "sensor_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String, ForeignKey("machines.id"), index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    cycle = Column(Integer, nullable=False, default=1)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # 15 NASA C-MAPSS core feature sensor readings
    s2 = Column(Float, default=640.0)
    s3 = Column(Float, default=1580.0)
    s4 = Column(Float, default=1400.0)
    s6 = Column(Float, default=21.6)
    s7 = Column(Float, default=553.0)
    s8 = Column(Float, default=2388.0)
    s9 = Column(Float, default=9050.0)
    s11 = Column(Float, default=47.5)
    s12 = Column(Float, default=521.0)
    s13 = Column(Float, default=2388.0)
    s14 = Column(Float, default=8120.0)
    s15 = Column(Float, default=8.4)
    s17 = Column(Float, default=392.0)
    s20 = Column(Float, default=38.8)
    s21 = Column(Float, default=23.2)

    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, default=0.0)

    machine = relationship("Machine", back_populates="sensor_readings")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, index=True)
    machine_id = Column(String, ForeignKey("machines.id"), index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    predicted_rul = Column(Float, nullable=False)
    rf_predicted_rul = Column(Float, nullable=False)
    model_used = Column(String, default="PyTorch LSTM")
    confidence_min = Column(Float, default=0.0)
    confidence_max = Column(Float, default=0.0)
    health_score = Column(Integer, default=100)
    risk_level = Column(String, default="healthy")  # healthy, warning, critical
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    machine = relationship("Machine", back_populates="predictions")

class AnomalyEvent(Base):
    __tablename__ = "anomalies"

    id = Column(String, primary_key=True, index=True)
    machine_id = Column(String, ForeignKey("machines.id"), index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    severity = Column(String, default="medium")  # low, medium, high, critical
    score = Column(Float, default=0.5)
    description = Column(Text, nullable=False)
    affected_sensors = Column(Text, default="[]")  # JSON encoded list of sensor names
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    machine = relationship("Machine", back_populates="anomalies")

class MaintenanceTask(Base):
    __tablename__ = "maintenance_records"

    id = Column(String, primary_key=True, index=True)
    machine_id = Column(String, ForeignKey("machines.id"), index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    title = Column(String, nullable=False)
    priority = Column(String, default="medium")  # low, medium, high, critical
    status = Column(String, default="open")      # open, in_progress, resolved
    description = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    machine = relationship("Machine", back_populates="maintenance_tasks")

class DiagnosticCase(Base):
    __tablename__ = "diagnostic_cases"

    id = Column(String, primary_key=True, index=True)
    machine_id = Column(String, ForeignKey("machines.id"), index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    trigger = Column(String, nullable=False)
    finding = Column(Text, nullable=False)
    explanation = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)
    severity = Column(String, default="medium")
    confidence = Column(Integer, default=85)
    generator_used = Column(String, default="Grounded Rule-Based Fallback")
    provider_used = Column(String, default="grounded_fallback")  # gemini, grok, grounded_fallback, ml_only
    fallback_level = Column(Integer, default=3)
    is_grounded_fallback = Column(Boolean, default=True)
    evidence = Column(Text, default="[]")  # JSON encoded list of evidence items
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    machine = relationship("Machine", back_populates="diagnostics")

class CopilotConversation(Base):
    __tablename__ = "copilot_conversations"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    machine_id = Column(String, ForeignKey("machines.id"), nullable=True)
    title = Column(String, default="Industrial Copilot Session")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    machine = relationship("Machine", back_populates="conversations")
    messages = relationship("CopilotMessage", back_populates="conversation", cascade="all, delete-orphan", order_by="CopilotMessage.created_at")

class CopilotMessage(Base):
    __tablename__ = "copilot_messages"

    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("copilot_conversations.id"), index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    sender = Column(String, nullable=False)  # user, copilot
    content = Column(Text, nullable=False)
    provider_used = Column(String, default="grounded_fallback")  # gemini, grok, grounded_fallback, ml_only, unavailable
    fallback_level = Column(Integer, default=1)
    evidence = Column(Text, default="[]")  # JSON list of retrieved chunks
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("CopilotConversation", back_populates="messages")

class KnowledgeDoc(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)  # Nullable: Null = SYSTEM_KNOWLEDGE
    scope = Column(String, default="SYSTEM_KNOWLEDGE")  # SYSTEM_KNOWLEDGE or USER_KNOWLEDGE
    title = Column(String, nullable=False)
    type = Column(String, default="PDF")
    size = Column(Integer, default=0)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    source_category = Column(String, default="Manual")
    indexed_chunks = Column(Integer, default=0)
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    page = Column(Integer, default=1)
    section = Column(String, default="Document")
    content = Column(Text, nullable=False)

    document = relationship("KnowledgeDoc", back_populates="chunks")
