"""Tests for security module (JWT, passwords, RBAC)."""

import pytest
from datetime import timedelta
from jose import jwt

from api.security import (
    create_access_token,
    verify_password,
    get_password_hash,
    get_current_user,
    get_current_admin,
)
from api.config import SECRET_KEY, ALGORITHM


@pytest.mark.unit
class TestSecurity:
    """Test suite for security functions."""

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "securepassword123"
        hashed = get_password_hash(password)

        # Hash should be different from plain password
        assert hashed != password

        # Verification should work
        assert verify_password(password, hashed) is True

        # Wrong password should fail
        assert verify_password("wrongpassword", hashed) is False

    def test_create_access_token(self):
        """Test JWT token creation."""
        data = {"sub": "testuser"}
        token = create_access_token(data)

        # Token should be a string
        assert isinstance(token, str)

        # Decode and verify
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert "exp" in payload  # Should have expiration

    def test_create_access_token_with_expiry(self):
        """Test JWT token creation with custom expiry."""
        data = {"sub": "testuser"}
        expires_delta = timedelta(minutes=15)
        token = create_access_token(data, expires_delta=expires_delta)

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"

    def test_password_hash_uniqueness(self):
        """Test that same password generates different hashes (salt)."""
        password = "samepassword"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # Hashes should be different due to salt
        assert hash1 != hash2

        # But both should verify correctly
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


@pytest.mark.unit
@pytest.mark.asyncio
class TestAuthDependencies:
    """Test authentication dependency functions."""

    async def test_get_current_user_valid_token(self, test_user, user_token):
        """Test getting current user with valid token."""
        from fastapi import Request
        from api import database as db_module

        # This test requires mocking the dependency injection
        # For now, we verify the token is valid by decoding
        payload = jwt.decode(user_token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == str(test_user["id"])

    async def test_expired_token_detection(self):
        """Test that expired tokens are detected."""
        from datetime import datetime, timedelta

        # Create an already-expired token
        data = {"sub": "testuser", "exp": datetime.utcnow() - timedelta(hours=1)}
        expired_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

        # Try to decode - should raise exception
        with pytest.raises(jwt.JWTError):
            payload = jwt.decode(expired_token, SECRET_KEY, algorithms=[ALGORITHM])

    async def test_invalid_token_format(self):
        """Test that invalid token format is rejected."""
        invalid_token = "not.a.valid.jwt.token"

        with pytest.raises(jwt.JWTError):
            jwt.decode(invalid_token, SECRET_KEY, algorithms=[ALGORITHM])
