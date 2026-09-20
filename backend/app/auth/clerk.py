import logging
import jwt
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.config import settings
from app.db.session import get_db
from app.db.models import User

logger = logging.getLogger("machinasense.auth")
security = HTTPBearer(auto_error=False)

# In-memory cache for Clerk JWKS keys
_jwks_client: Optional[jwt.PyJWKClient] = None

def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Verifies Clerk authentication token and returns the current authenticated User model.
    Enforces strict user identification and auto-provisions user row in database.
    """
    # Identity is never accepted from a browser-provided user_id.
    token = None
    if credentials:
        token = credentials.credentials
    else:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. Decode and verify Clerk JWT
    try:
        # Deliberately narrow test seam; disabled outside the test environment.
        if token.startswith("test_token_") and settings.ENVIRONMENT == "test":
            user_id = token.replace("test_token_", "")
            return _get_or_create_user(db, user_id, f"{user_id}@machinasense.test")

        if not settings.CLERK_JWKS_URL:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clerk JWT verification is not configured.")
        global _jwks_client
        if _jwks_client is None:
            _jwks_client = jwt.PyJWKClient(settings.CLERK_JWKS_URL)
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        header = jwt.get_unverified_header(token)
        payload = jwt.decode(token, signing_key.key, algorithms=[header.get("alg", "RS256")], issuer=settings.CLERK_ISSUER or None, options={"verify_aud": False, "verify_iss": bool(settings.CLERK_ISSUER)})
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject claim.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        email = (
            payload.get("email") or
            payload.get("primary_email_address") or
            f"{user_id}@user.machinasense.io"
        )
        first_name = payload.get("first_name", "")
        last_name = payload.get("last_name", "")

        return _get_or_create_user(db, user_id, email, first_name, last_name)

    except jwt.PyJWTError as e:
        logger.warning(f"JWT verification failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

def _get_or_create_user(
    db: Session,
    user_id: str,
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None
) -> User:
    """Ensures user exists in the local database."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(
            id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Provisioned new user record for {user_id} ({email}).")
    return user
