"""
Engine/session plumbing, kept separate from models so tests can spin up a
throwaway in-memory database without touching the real one.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            # a plain in-memory sqlite db is per-connection -- without a
            # shared StaticPool, every new session would see an empty,
            # table-less database. Only relevant for tests.
            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """FastAPI dependency: yields a session, always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(engine_=None):
    """Create all tables. Called at app startup; also used directly by tests."""
    Base.metadata.create_all(bind=engine_ or engine)
