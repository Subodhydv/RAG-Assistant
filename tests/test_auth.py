"""
Unit tests for HMAC session token signing, token verification, and tampering prevention.
"""
import pytest
from fastapi import HTTPException

from app.auth import sign_session_id, verify_session_token, verify_credentials


def test_sign_and_verify_session_token():
    raw_id = "sess_user_99"
    signed_token = sign_session_id(raw_id)
    assert "." in signed_token
    assert signed_token.startswith("sess_user_99.")

    verified_id = verify_session_token(signed_token)
    assert verified_id == "sess_user_99"


def test_tampered_session_token_rejection():
    raw_id = "sess_user_99"
    signed_token = sign_session_id(raw_id)

    # Tamper with the session ID part
    tampered_token = "sess_other_user." + signed_token.split(".")[1]
    
    with pytest.raises(HTTPException) as exc_info:
        verify_session_token(tampered_token)
    
    assert exc_info.value.status_code == 401
    assert "signature mismatch" in exc_info.value.detail.lower()


def test_verify_credentials():
    assert verify_credentials("admin", "admin123") is True
    assert verify_credentials("admin", "wrongpassword") is False
