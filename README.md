# MachinaSense

MachinaSense is an authenticated industrial predictive-maintenance portfolio platform. It combines user-owned machine telemetry with the existing NASA C-MAPSS FD001 ML assets, a private technical-document knowledge base, and an engineering copilot.

## Architecture

- React/Vite frontend protected by Clerk (`/sign-in`, `/sign-up` and protected application routes).
- FastAPI API verifies Clerk JWTs against the configured JWKS; frontend identity fields are never trusted.
- SQLAlchemy persistence, configured for PostgreSQL in production (`DATABASE_URL`); local SQLite can be used only for development/testing.
- Every machine, telemetry row, prediction, anomaly, document, conversation and message carries server-side user ownership.
- CSV ingestion validates file type/size, required 15 C-MAPSS feature columns, missing values, malformed rows and numeric values before persistence and ML inference.
- Existing Random Forest, PyTorch LSTM, scaler, sequence window, RUL cap, Isolation Forest and artifacts remain system assets.
- User documents support PDF, DOCX and TXT extraction/chunking with provenance. They remain private to the uploader.
- Copilot provider chain: **Gemini → Grok → Grounded Deterministic RAG → ML-only → unavailable**. Gemini and Grok keys stay backend-only.

## Benchmark limitation

The Random Forest (MAE ≈ 13.22, RMSE ≈ 18.10, R² ≈ 0.8102) and LSTM (MAE ≈ 11.30, RMSE ≈ 15.00, R² ≈ 0.8697) results are **NASA C-MAPSS FD001 benchmark results**, not validated industrial-deployment performance. Isolation Forest remains unsupervised with contamination `0.05`.

## Local setup

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
cd backend
..\venv\Scripts\Activate.ps1
pip install -r requirements.txt
alembic upgrade head
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

## Required production configuration

Backend (`backend/.env`):

```ini
DATABASE_URL=postgresql+psycopg2://...
CLERK_SECRET_KEY=
CLERK_ISSUER=https://...
CLERK_JWKS_URL=https://.../.well-known/jwks.json
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GROK_API_KEY=
GROK_MODEL=grok-3-mini
CORS_ORIGINS=https://your-site.netlify.app
ENVIRONMENT=production
LOG_LEVEL=INFO
```

Frontend (`frontend/.env`):

```ini
VITE_API_BASE_URL=https://your-fastapi-host.example
VITE_CLERK_PUBLISHABLE_KEY=pk_...
```

Never expose backend values, especially Clerk secret, database URL, Gemini key or Grok key, in Vite variables or a committed `.env` file.

## Deployment

`netlify.toml` builds from `frontend` and serves `dist`, with an SPA rewrite so direct URLs such as `/machines/:id`, `/diagnostics/:id` and `/copilot` work. Set Netlify’s `VITE_API_BASE_URL` to the deployed FastAPI origin, not localhost. Explicitly add the Netlify origin to backend `CORS_ORIGINS`.

`GET /health` stays public; application APIs require a valid Clerk session.

## Verification

```powershell
.\venv\Scripts\python.exe -m pytest backend\tests -q
cd frontend
npm run lint
npm run build
```

The test suite covers protected routes, invalid authentication, cross-user access, document isolation, malformed telemetry, and the full LLM fallback sequence.
