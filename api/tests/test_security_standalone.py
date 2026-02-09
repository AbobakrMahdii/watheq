"""Standalone security tests with path fix."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import timedelta


def test_password_hashing():
    """Test password hashing without any fixtures."""
    from api.security import get_password_hash, verify_password
    
    password = "testpassword123"
    hashed = get_password_hash(password)
    
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False
    print("✓ Password hashing test passed")


def test_jwt_token_creation():
    """Test JWT token creation without fixtures."""
    from api.security import create_access_token
    from jose import jwt
    from api.config import SECRET_KEY, ALGORITHM
    
    data = {"sub": "testuser"}
    token = create_access_token(data)
    
    assert isinstance(token, str)
    
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "testuser"
    assert "exp" in payload
    print("✓ JWT token creation test passed")


def test_password_uniqueness():
    """Test that password hashing uses salt."""
    from api.security import get_password_hash, verify_password
    
    password = "samepass"
    hash1 = get_password_hash(password)
    hash2 = get_password_hash(password)
    
    assert hash1 != hash2
    assert verify_password(password, hash1) is True
    assert verify_password(password, hash2) is True
    print("✓ Password hash uniqueness test passed")


if __name__ == "__main__":
    print("Running standalone security tests...")
    print()
    test_password_hashing()
    test_jwt_token_creation()
    test_password_uniqueness()
    print()
    print("=" * 50)
    print("✓ All 3 backend security tests passed!")
    print("=" * 50)
