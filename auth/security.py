"""Password hashing and JWT token codecs — swappable strategies."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from abc import ABC, abstractmethod

from config import AuthConfig

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    from passlib.context import CryptContext as _CryptContext

    _HAS_PASSLIB = True
except ImportError:
    _CryptContext = None
    _HAS_PASSLIB = False

try:  # pragma: no cover - environment dependent
    from jose import jwt as _jose_jwt

    _HAS_JOSE = True
except ImportError:
    _jose_jwt = None
    _HAS_JOSE = False
    logger.info("python-jose unavailable — using stdlib HS256 JWT codec")


class PasswordHasher(ABC):
    @abstractmethod
    def hash(self, password: str) -> str: ...

    @abstractmethod
    def verify(self, password: str, hashed: str) -> bool: ...


class BcryptHasher(PasswordHasher):
    """passlib + bcrypt (preferred when installed)."""

    name = "bcrypt"

    def __init__(self) -> None:
        self._context = _CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash(self, password: str) -> str:
        return self._context.hash(password)

    def verify(self, password: str, hashed: str) -> bool:
        return self._context.verify(password, hashed)


class Pbkdf2Hasher(PasswordHasher):
    """Stdlib fallback: PBKDF2-HMAC-SHA256, format pbkdf2_sha256$iter$salt$hash."""

    name = "pbkdf2_sha256"
    iterations = 200_000

    def hash(self, password: str) -> str:
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), self.iterations)
        return f"{self.name}${self.iterations}${salt}${digest.hex()}"

    def verify(self, password: str, hashed: str) -> bool:
        try:
            scheme, iterations, salt, expected = hashed.split("$", 3)
            if scheme != self.name:
                return False
            digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
            return hmac.compare_digest(digest.hex(), expected)
        except (ValueError, TypeError):
            return False


class TokenCodec(ABC):
    @abstractmethod
    def encode(self, payload: dict) -> str: ...

    @abstractmethod
    def decode(self, token: str) -> dict: ...


class JoseJwtCodec(TokenCodec):
    """python-jose implementation (preferred when installed)."""

    def __init__(self, config: AuthConfig) -> None:
        self.config = config

    def encode(self, payload: dict) -> str:
        return _jose_jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

    def decode(self, token: str) -> dict:
        try:
            return _jose_jwt.decode(
                token, self.config.secret_key,
                algorithms=[self.config.algorithm],
            )
        except Exception as err:
            raise ValueError(f"invalid token: {err}") from err


class HmacJwtCodec(TokenCodec):
    """Minimal stdlib JWT (HS256) — same wire format as python-jose output."""

    def __init__(self, config: AuthConfig) -> None:
        self.config = config

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    @staticmethod
    def _unb64(data: str) -> bytes:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))

    def encode(self, payload: dict) -> str:
        header = self._b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        body = self._b64(json.dumps(payload).encode())
        signature = hmac.new(
            self.config.secret_key.encode(), f"{header}.{body}".encode(), hashlib.sha256
        ).digest()
        return f"{header}.{body}.{self._b64(signature)}"

    def decode(self, token: str) -> dict:
        header, body, signature = token.split(".")
        expected = hmac.new(
            self.config.secret_key.encode(), f"{header}.{body}".encode(), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(self._b64(expected), signature):
            raise ValueError("invalid token signature")
        payload = json.loads(self._unb64(body))
        if "exp" in payload and payload["exp"] < time.time():
            raise ValueError("token expired")
        return payload


def build_password_hasher() -> PasswordHasher:
    if _HAS_PASSLIB:
        try:
            hasher = BcryptHasher()
            hasher.hash("probe")
            return hasher
        except Exception:
            logger.info("passlib bcrypt backend unavailable or incompatible — using stdlib PBKDF2 hasher")
            return Pbkdf2Hasher()
    return Pbkdf2Hasher()


def build_token_codec(config: AuthConfig) -> TokenCodec:
    return JoseJwtCodec(config) if _HAS_JOSE else HmacJwtCodec(config)