"""Simple security tests without database dependencies."""

import pytest
from datetime import timedelta

from api.security import (
    create_access_token,
    verify_password,
    get_password_hash,
)


class TestPasswordHashing:
    """Test password hashing functions."""
    
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


class TestJWT:
    """Test JWT token functions."""
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        from jose import jwt
        from api.config import SECRET_KEY, ALGORITHM
        
        data = {"sub": "testuser"}
        token = create_access_token(data)
        
        # Token should be a string
        assert isinstance(token, str)
        
        # Decode and verify
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert "exp" in payload  # Should have expiration
    
    def test_create_access_token_with_custom_expiry(self):
        """Test JWT token creation with custom expiry."""
        from jose import jwt
        from api.config import SECRET_KEY, ALGORITHM
        
        data = {"sub": "testuser"}
        expires_delta = timedelta(minutes=15)
        token = create_access_token(data, expires_delta=expires_delta)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert "exp" in payload
