"""
Authentication and Authorization module for RAG Teaching Assistant.

Provides HMAC session token signing, verification, and FastAPI security dependencies.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import uuid
from typing import Optional

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings

security = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.now(datetime.timezone.utc) + (
        expires_delta or datetime.timedelta(hours=24)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm="HS256")
    return encoded_jwt


def verify_credentials(username: str, password: str) -> bool:
    return (
        username == settings.auth_username
        and password == settings.auth_password
    )


def sign_session_id(raw_id: str) -> str:
    clean_id = "".join(c for c in raw_id if c.isalnum() or c in ("-", "_"))[:32] or "default"
    sig = hmac.new(
        settings.session_secret_key.encode("utf-8"),
        clean_id.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()[:16]
    return f"{clean_id}.{sig}"


def verify_session_token(token: str) -> str:
    clean_token = token.strip()
    if not clean_token:
        return "default"
    if "." not in clean_token:
        # Graceful fallback for legacy or plain session IDs
        return "".join(c for c in clean_token if c.isalnum() or c in ("-", "_"))[:32] or "default"

    parts = clean_token.rsplit(".", 1)
    raw_id, sig = parts[0], parts[1]
    clean_raw_id = "".join(c for c in raw_id if c.isalnum() or c in ("-", "_"))[:32] or "default"

    expected_sig = hmac.new(
        settings.session_secret_key.encode("utf-8"),
        clean_raw_id.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()[:16]

    if not hmac.compare_digest(sig, expected_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token signature mismatch or unauthorized session access"
        )

    return clean_raw_id


def get_current_session(x_session_id: Optional[str] = Header(None, alias="X-Session-ID")) -> str:
    if not x_session_id or not x_session_id.strip():
        return "default"
    return verify_session_token(x_session_id.strip())


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    if not settings.auth_enabled:
        return {"sub": "guest", "role": "admin"}

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
