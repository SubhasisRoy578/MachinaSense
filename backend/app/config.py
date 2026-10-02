import os
from typing import List

class Settings:
    PROJECT_NAME: str = "MachinaSense Industrial Intelligence & Grounded RAG API"
    VERSION: str = "3.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
    DEBUG: bool = ENVIRONMENT == "development"
    
    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # CORS configuration
    _cors_env: str = os.getenv("CORS_ORIGINS", "")
    @property
    def CORS_ORIGINS(self) -> List[str]:
        raw = self._cors_env or os.getenv("CORS_ORIGIN", "")
        if not raw:
            return ["http://localhost:5173", "http://localhost:3000"] if self.DEBUG else ["https://machinasense.netlify.app"]
        origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
        if self.ENVIRONMENT == "production":
            # Strict production CORS: reject wildcard '*'
            origins = [o for o in origins if o != "*"]
            if not origins:
                origins = ["https://machinasense.netlify.app"]
        elif "*" in origins and self.ENVIRONMENT == "development":
            return ["*"]
        return origins

    # Database Configuration (PostgreSQL with SQLite local dev/test fallback)
    _db_url: str = os.getenv("DATABASE_URL", "")
    @property
    def DATABASE_URL(self) -> str:
        url = self._db_url or os.getenv("DATABASE_URL", "")
        if not url:
            if self.ENVIRONMENT == "production":
                raise RuntimeError("DATABASE_URL must be configured in production. Ephemeral SQLite fallback is not permitted in production mode.")
            # Fallback to local SQLite if DATABASE_URL is not set
            return "sqlite:///./machinasense.db"
        # Fix Heroku/Supabase postgres:// scheme
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql://", 1)
        return url

    # Clerk Authentication
    CLERK_SECRET_KEY: str = os.getenv("CLERK_SECRET_KEY", "")
    CLERK_ISSUER: str = os.getenv("CLERK_ISSUER", "")
    CLERK_JWKS_URL: str = os.getenv("CLERK_JWKS_URL", "")

    # LLM API Keys (Gemini Primary, Grok Secondary)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    GROK_API_KEY: str = os.getenv("GROK_API_KEY") or ""
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-3-mini")

    # Upload Limits & Validation
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt"}
    TELEMETRY_ALLOWED_EXTENSIONS: set = {".csv"}

settings = Settings()
