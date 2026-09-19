import os
from typing import List

class Settings:
    PROJECT_NAME: str = "MachinaSense Industrial Intelligence & Grounded RAG API"
    VERSION: str = "2.1.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
    DEBUG: bool = ENVIRONMENT == "development"
    
    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # CORS configuration
    _cors_env: str = os.getenv("CORS_ORIGINS", "*")
    @property
    def CORS_ORIGINS(self) -> List[str]:
        if self._cors_env == "*":
            return ["*"]
        return [origin.strip() for origin in self._cors_env.split(",") if origin.strip()]

    # Database Configuration (PostgreSQL with SQLite local dev/test fallback)
    _db_url: str = os.getenv("DATABASE_URL", "")
    @property
    def DATABASE_URL(self) -> str:
        if not self._db_url:
            # Fallback to local SQLite if DATABASE_URL is not set
            return "sqlite:///./machinasense.db"
        # Fix Heroku/Supabase postgres:// scheme
        if self._db_url.startswith("postgres://"):
            return self._db_url.replace("postgres://", "postgresql://", 1)
        return self._db_url

    # Clerk Authentication
    CLERK_SECRET_KEY: str = os.getenv("CLERK_SECRET_KEY", "")
    CLERK_PUBLISHABLE_KEY: str = os.getenv("CLERK_PUBLISHABLE_KEY", "") or os.getenv("VITE_CLERK_PUBLISHABLE_KEY", "")

    # LLM API Keys (Gemini Primary, Grok Secondary)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    GROK_API_KEY: str = os.getenv("GROK_API_KEY") or ""

    # Upload Limits & Validation
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt"}

settings = Settings()
