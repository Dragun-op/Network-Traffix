"""
Centralized configuration. Nothing in the rest of the app should read an
env var directly -- everything goes through `settings` so we have one
place to see (and override in tests) every knob the backend has.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Database ---
    # Defaults to a local SQLite file so any teammate can `fastapi dev main.py`
    # with zero setup. Swap DATABASE_URL in .env for Postgres in deployment.
    database_url: str = "sqlite:///./network_traffix.db"

    # --- CORS ---
    # The frontend is opened as a static file / separate dev server, so the
    # browser origin will never match the API origin -- CORS must be explicit.
    cors_origins: list[str] = [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "null",  # covers dashboard.html opened directly via file://
    ]

    # --- Inference / ML artifacts ---
    # Directory the ML teammate's train.py writes model.pkl / features.json /
    # threshold.json / shap_explainer.pkl into. If these files aren't present
    # yet, the backend automatically falls back to a rule-based scorer (see
    # app/inference/model.py) so nobody is blocked waiting on the ML side.
    model_artifacts_dir: str = "./ml_artifacts"
    decision_threshold: float = 0.5

    # --- Replay worker (our stand-in for "real-time" traffic) ---
    replay_enabled: bool = True
    replay_interval_seconds: float = 3.0

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NT_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
