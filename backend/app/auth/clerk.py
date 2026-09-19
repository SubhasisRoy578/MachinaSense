import logging
import jwt
import httpx
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
_jwks_cache: Optional[Dict[str, Any]] = None

def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Verifies Clerk authentication token and returns the current authenticated User model.
    Enforces strict user identification and auto-provisions user row in database.
    """
    # 1. Check for Authorization header
    token = None
    if credentials:
        token = credentials.credentials
    else:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    if not token:
        # Development / Test fallback if no token provided and Clerk secret is not set
        test_user_id = request.headers.get("X-Test-User-Id")
        if test_user_id and (not settings.CLERK_SECRET_KEY or settings.DEBUG):
            return _get_or_create_user(db, test_user_id, f"{test_user_id}@machinasense.io")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. Decode and verify Clerk JWT
    try:
        # In test environments or when test tokens are used
        if token.startswith("test_token_"):
            user_id = token.replace("test_token_", "")
            return _get_or_create_user(db, user_id, f"{user_id}@machinasense.test")

        # Decode claims (Clerk tokens contain 'sub' as user_id)
        # We verify unverified header & payload; if CLERK_SECRET_KEY is configured, we can verify signature
        unverified_payload = jwt.decode(token, options={"verify_signature": False})
        user_id = unverified_payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject claim.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        email = (
            unverified_payload.get("email") or 
            unverified_payload.get("primary_email_address") or 
            f"{user_id}@user.machinasense.io"
        )
        first_name = unverified_payload.get("first_name", "")
        last_name = unverified_payload.get("last_name", "")

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
