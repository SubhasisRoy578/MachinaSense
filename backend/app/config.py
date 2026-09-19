import os
from typing import List

class Settings:
    PROJECT_NAME: str = "MachinaSense Industrial Intelligence & Grounded RAG API"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
    DEBUG: bool = ENVIRONMENT == "development"
    
    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # CORS configuration
    # Accepts comma-separated list of origins, e.g. "http://localhost:5173,http://localhost:3000"
    _cors_env: str = os.getenv("CORS_ORIGINS", "*")
    @property
    def CORS_ORIGINS(self) -> List[str]:
        if self._cors_env == "*":
            return ["*"]
        return [origin.strip() for origin in self._cors_env.split(",") if origin.strip()]

    # LLM API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or ""

    # Upload Limits & Validation
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt"}

settings = Settings()
