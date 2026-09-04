"""SQLAlchemy engine/session management for SQLite."""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import DatabaseConfig
from db.models import Base

logger = logging.getLogger(__name__)


class Database:
    """Owns the engine and session factory; creates the schema on init."""

    def __init__(self, config: DatabaseConfig) -> None:
        if config.url.startswith("sqlite:///"):
            db_file = config.url.removeprefix("sqlite:///")
            if db_file != ":memory:":
                Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(config.url, echo=config.echo, connect_args={"check_same_thread": False}
                                    if config.url.startswith("sqlite") else {})
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        logger.info("Database schema created")

    def session(self) -> Session:
        return self._session_factory()

    def drop_schema(self) -> None:
        Base.metadata.drop_all(self.engine)


def get_session_factory(config: DatabaseConfig):
    """Convenience for tests: returns (Database, session_factory)."""
    database = Database(config)
    database.create_schema()
    return database, database._session_factory