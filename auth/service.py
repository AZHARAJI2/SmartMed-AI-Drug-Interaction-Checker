"""AuthService — patient registration, login, JWT issuing (patients only, no other roles)."""
from __future__ import annotations

import logging
import time

from auth.security import PasswordHasher, TokenCodec
from config import AuthConfig
from db.models import User
from db.repositories import UserRepository

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, user_repo: UserRepository, hasher: PasswordHasher,
                 codec: TokenCodec, config: AuthConfig) -> None:
        self.users = user_repo
        self.hasher = hasher
        self.codec = codec
        self.config = config

    def register(self, name: str, email: str, password: str) -> User:
        if len(password) < 6:
            raise ValueError("password must be at least 6 characters")
        if self.users.get_by_email(email) is not None:
            raise ValueError("email already registered")
        user = self.users.add(name, email, self.hasher.hash(password))
        logger.info("Registered patient %s (id=%d)", email, user.id)
        return user

    def authenticate(self, email: str, password: str) -> tuple[User, str]:
        """Returns (user, jwt) or raises ValueError on bad credentials."""
        user = self.users.get_by_email(email)
        if user is None or not self.hasher.verify(password, user.password_hash):
            raise ValueError("invalid email or password")
        return user, self.create_token(user)

    def create_token(self, user: User) -> str:
        payload = {
            "sub": str(user.id),
            "email": user.email,
            "role": "patient",
            "exp": int(time.time()) + self.config.token_expiry_minutes * 60,
        }
        return self.codec.encode(payload)

    def resolve_token(self, token: str) -> User | None:
        """Decode a JWT and return the patient, or None when invalid/expired."""
        try:
            payload = self.codec.decode(token)
        except ValueError:
            return None
        user = self.users.get(int(payload.get("sub", 0)))
        if user is None or payload.get("role") != "patient":
            return None
        return user