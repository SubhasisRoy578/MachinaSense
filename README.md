# MachinaSense — Industrial Machine Intelligence & Grounded AI Diagnostic Platform

MachinaSense is a production-grade, AI-powered predictive maintenance and machine intelligence platform engineered for turbofan jet engine fleets. It combines real deep learning and regression models trained on run-to-failure telemetry from **NASA C-MAPSS FD001** with a grounded **Retrieval-Augmented Generation (RAG)** industrial diagnostic engine, served via a hardened FastAPI backend and a high-performance React 19 operational control center.

---

## Architecture

```text
       ┌─────────────────────────────────────────────────────────┐
       │               NASA C-MAPSS FD001 Dataset                │
       │  (100 Train Units / 100 Test Units / 15 Sensor Signals) │
       └────────────────────────────┬────────────────────────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │             ML Preprocessing & Normalization            │
       │    (Unit-Aware Splitting, MinMaxScaler, 30-Cycle Window)│
       └───────┬────────────────────┬────────────────────┬───────┘
               ▼                    ▼                    ▼
     ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
     │  Random Forest   │ │   PyTorch LSTM   │ │ Isolation Forest │
     │   RUL Baseline   │ │   Temporal RUL   │ │ Anomaly Detector │
     │  (MAE: 13.22)    │ │   (MAE: 11.30)   │ │  (15 Sensors)    │
     └─────────┬────────┘ └────────┬─────────┘ └────────┬─────────┘
               └────────────────────┼────────────────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │              Hardened FastAPI REST Gateway              │
       │  (CORS, Structured Logging, Sanitized Uploads, Pydantic)│
       └───────┬─────────────────────────────────────────┬───────┘
               ▼                                         ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│  React 19 Operational Hub    │        │  Grounded RAG Engine         │
│  - Route Code-Splitting      │        │  - PDF/DOCX/TXT Extractor    │
│  - 5 System Status Modes     │        │  - Semantic Chunker (Page ID)│
│  - Dual Provider (Live/Mock) │        │  - TF-IDF Vector Store       │
│  - Real-Time Fleet Telemetry │        │  - Gemini/OpenAI Synthesis   │
│  - Interactive Diagnostics   │◄───────┤  - Grounded Fallback Engine  │
└──────────────────────────────┘        └──────────────────────────────┘
```

---

## 1. Real Machine Learning Models & Evaluation

Models were trained and evaluated on unseen NASA C-MAPSS FD001 test units using ground-truth Remaining Useful Life (RUL) labels:

| Model Architecture | Task | Key Hyperparameters | Test MAE | Test RMSE | Test R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch LSTM** | Temporal RUL Prediction | 2-layer LSTM, hidden size 64, dropout 0.2, 30-cycle sequence window | **11.30 cycles** | **15.00 cycles** | **0.8697** |
| **Random Forest** | RUL Baseline | 100 estimators, max depth 10, random state 42 | **13.22 cycles** | **18.10 cycles** | **0.8102** |
| **Isolation Forest** | Multivariate Sensor Anomaly Detection | Contamination: 0.05, 100 estimators, unsupervised on 15 core sensor channels | *Unsupervised* | *N/A* | *N/A* |

*Note: Isolation Forest anomaly detection is strictly unsupervised as FD001 does not include synthetic ground-truth anomaly labels.*

### Model Artifacts (`backend/artifacts/`)
- `backend/artifacts/rul/lstm_model.pth` — PyTorch LSTM weights
- `backend/artifacts/rul/rf_baseline.joblib` — Random Forest regressor
- `backend/artifacts/anomaly/isolation_forest.joblib` — Isolation Forest model
- `backend/artifacts/preprocessing/scaler.joblib` — Fitted feature normalization scaler
- `backend/artifacts/model_metadata.json` — Training parameters and sensor mappings
- `backend/artifacts/evaluation.json` — Rigorous evaluation benchmarks

---

## 2. Grounded RAG & AI Diagnostics

MachinaSense bridges numerical ML predictions with technical maintenance literature through grounded Retrieval-Augmented Generation:

- **Document Extraction (`backend/app/rag/extractor.py`)**: Validates and extracts text from PDF (`pypdf`), Word (`python-docx`), and plain text (`txt`) manuals up to 25MB with filename sanitization against path traversal.
- **Semantic Chunking (`backend/app/rag/chunker.py`)**: Partitions technical manuals into 500-character segments with 100-character overlaps, preserving document ID, section titles, and physical page provenance.
- **Vector Retrieval (`backend/app/rag/vector_store.py`)**: Disk-persisted TF-IDF vector index and cosine similarity search engine (`vector_index.json` & `documents_catalog.json`).
- **Grounded Diagnostic Engine (`backend/app/rag/llm_service.py`)**:
  - **Live LLM Synthesis**: Supports Google Gemini 1.5 Flash and OpenAI GPT-4o Mini when API keys are present.
  - **Deterministic Grounded Fallback**: Automatically activates when no API key is provided. Uses technical manual chunks and ML findings deterministically. Explicitly tagged with `isGroundedFallback: true` and labeled in the UI as **Grounded Fallback** (never fabricated as an LLM response).

---

## 3. Operational System States

The platform dynamically detects and displays 5 operational modes across headers, diagnostics, and settings:

| Status Badge | Mode Key | Backend State | Description |
| :--- | :--- | :--- | :--- |
| **LIVE ML + REAL RAG** | `live_rag` | Connected | FastAPI active + ML models loaded + LLM API key configured. Full neural RAG synthesis. |
| **LIVE ML + GROUNDED FALLBACK** | `live_fallback` | Connected | FastAPI active + ML models loaded + TF-IDF retrieval active (No external LLM key). Deterministic grounded fallback active. |
| **LIVE ML (RAG UNAVAILABLE)** | `live_no_rag` | Connected | FastAPI active + ML models loaded + Vector index empty. |
| **DEVELOPMENT MOCK** | `mock` | Standalone | Frontend running against local mock data generator without backend network requirements. |
| **BACKEND UNREACHABLE** | `error` | Disconnected | Backend configured but server unreachable. Seamlessly falls back to mock provider with warning banner. |

---

## 4. Frontend Architecture & Verified Routes

The React 19 dashboard uses Vite, Tailwind CSS, Lucide icons, and Recharts. All routes use **`React.lazy()` and `Suspense` route-based code splitting**, keeping individual chunk sizes well under 360 kB.

### 12 Verified Routes:
1. `/` — Operations Overview (Fleet health KPIs, health distribution, telemetry stream)
2. `/machines` — Machine Fleet Inventory (Status filtering, RUL, failure risk)
3. `/machines/:id` — Machine Detail Telemetry (Area charts, LSTM RUL forecast curve, active anomalies, Run Grounded AI Diagnostic button)
4. `/sensors` — Sensor Intelligence (Fleet sensor KPIs, dynamic machine selection from live fleet)
5. `/anomalies` — Anomaly Event Center (Isolation Forest events timeline, sensor breakdown)
6. `/predictions` — RUL Forecast Center (LSTM vs RF comparison, cycle degradation)
7. `/maintenance` — Maintenance Schedule & Work Orders
8. `/diagnostics` — AI Diagnostics Center (Investigation cases, generator badge column, fallback filter)
9. `/diagnostics/:id` — Diagnostic Investigation View (Grounded explanation, evidence with page provenance, fallback alerts)
10. `/knowledge-base` — Technical Knowledge Base (Real drag-and-drop file upload, document catalog, TF-IDF vector search)
11. `/analytics` — ML Model Performance & NASA C-MAPSS Metrics
12. `/settings` — Production System Configuration & Health Dashboard

---

## 5. FastAPI REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health readiness, models loaded, RAG status, LLM provider |
| `GET` | `/api/machines` | List all monitored C-MAPSS turbofan engines |
| `GET` | `/api/machines/{id}` | Detailed telemetry and operational status for an engine |
| `GET` | `/api/machines/{id}/sensors` | 24-cycle time-series sensor history |
| `GET` | `/api/machines/{id}/prediction` | PyTorch LSTM and Random Forest RUL predictions & confidence intervals |
| `GET` | `/api/machines/{id}/anomalies` | Isolation Forest anomaly event timeline |
| `GET` | `/api/machines/{id}/health` | Machine health index trend |
| `GET` | `/api/analytics/models` | ML training metadata and evaluation benchmarks |
| `POST` | `/api/predict/rul` | Real-time RUL prediction on custom 15-sensor inputs or sequence matrices |
| `POST` | `/api/detect/anomaly` | Real-time anomaly detection on custom sensor inputs |
| `GET` | `/api/knowledge/documents` | List indexed technical manuals in the vector catalog |
| `GET` | `/api/knowledge/documents/{id}` | Get metadata for a specific document |
| `GET` | `/api/knowledge/documents/{id}/chunks` | List semantic chunks with page provenance |
| `POST` | `/api/knowledge/upload` | Upload and index technical manuals (`.pdf`, `.docx`, `.txt`) |
| `POST` | `/api/knowledge/search` | TF-IDF vector similarity search against technical manual chunks |
| `POST` | `/api/diagnostics` | Generate grounded AI diagnostic investigation case |
| `GET` | `/api/diagnostics` | List all diagnostic investigation cases |
| `GET` | `/api/diagnostics/{id}` | Get diagnostic case details with supporting evidence |

---

## 6. Local Setup & Execution

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Node.js 18+ & npm

### Backend Setup
```powershell
# 1. Navigate to backend directory
cd backend

# 2. Activate Python virtual environment
..\venv\Scripts\Activate.ps1    # Windows PowerShell
# source ../venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation: `http://localhost:8000/docs`

### Frontend Setup
```powershell
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev
```
Application UI: `http://localhost:5173`

---

## 7. Environment Variables Configuration

Copy `.env.example` to `.env`:

```ini
# --- Backend Configuration ---
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173

# --- Optional LLM API Keys (Grounded Fallback used if omitted) ---
# GEMINI_API_KEY=your_gemini_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here

# --- Frontend Configuration ---
VITE_API_BASE_URL=http://localhost:8000
```

---

## 8. Docker Deployment Preparation

A production multi-container setup is prepared via Docker and Docker Compose:

```powershell
# Build and run complete stack (FastAPI Backend + Nginx Frontend)
docker-compose up --build -d

# Check container health
docker-compose ps
```

- **Backend Container (`backend/Dockerfile`)**: Multi-stage `python:3.11-slim` image with healthcheck.
- **Frontend Container (`frontend/Dockerfile`)**: Multi-stage build on `node:20-alpine`, served by `nginx:alpine` with SPA routing and gzip compression.

---

## 9. Comprehensive Testing & Verification Results

### Backend Automated Test Suite
Run all 33 automated tests across configuration, machine APIs, prediction, RAG, and hardening:

```powershell
# Run from backend directory
python -m pytest
```

**Result: 33 passed in 69.17s**
- `tests/test_api.py` (8 tests): Health, machine list, machine details, 404 handling, sensor telemetry, RUL inference, anomaly detection.
- `tests/test_rag.py` (9 tests): Text extraction, empty file validation, chunk provenance, vector search, RAG diagnostic generation, insufficient evidence handling, knowledge endpoints.
- `tests/test_hardening.py` (16 tests): Config defaults, filename sanitization / traversal protection, detailed health schema, invalid 404s, input dimension validation (15 features), malicious upload prevention, empty query rejection.

### Frontend Type Check & Production Build
```powershell
cd frontend
npx tsc -b
npm run build
```

**Result:**
- **TypeScript (`tsc -b`)**: **0 errors**.
- **Vite Build (`npm run build`)**: **Success in 24.00s**. All routes lazy-loaded; largest vendor chunk is only 353 kB with zero bundle size warnings.
