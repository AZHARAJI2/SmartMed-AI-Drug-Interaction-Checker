"""Auth package: JWT + bcrypt-style hashing, patients only.

python-jose/passlib are used when installed; stdlib HS256-HMAC and PBKDF2
implementations keep the system fully functional offline ( interchangeable
strategies behind ABCs, per the Master plan's OOP commitment).
"""
from auth.security import (
    PasswordHasher,
    TokenCodec,
    Pbkdf2Hasher,
    HmacJwtCodec,
    build_password_hasher,
    build_token_codec,
)
from auth.service import AuthService

__all__ = [
    "PasswordHasher",
    "TokenCodec",
    "Pbkdf2Hasher",
    "HmacJwtCodec",
    "build_password_hasher",
    "build_token_codec",
    "AuthService",
]