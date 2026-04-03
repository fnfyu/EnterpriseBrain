import pytest
from app.core.security import hash_password, verify_password, create_access_token, decode_token
from app.core.config import get_settings

settings = get_settings()


def test_password_hashing():
    pw = "SecurePass123!"
    hashed = hash_password(pw)
    assert hashed != pw
    assert verify_password(pw, hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_roundtrip():
    token = create_access_token({"sub": "user-123", "username": "alice"})
    payload = decode_token(token)
    assert payload["sub"] == "user-123"
    assert payload["username"] == "alice"


def test_invalid_token_raises():
    with pytest.raises(ValueError):
        decode_token("not.a.valid.token")
