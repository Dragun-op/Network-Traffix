"""
register_model.py -- write a ModelVersion row from the artifacts train.py
produced, so GET /api/metrics reports the real precision/recall/FPR instead of
the placeholder zeros.

metrics_service.get_current_metrics() reads the latest ModelVersion row; this is
the step that creates it. Safe to run repeatedly -- it skips if a row with the
same trained_at already exists.

Run after training:
    cd backend && python -m app.register_model
    #   --artifacts-dir ./ml_artifacts   (defaults to settings.model_artifacts_dir)

Or call register_from_artifacts() from main.py's lifespan to make it automatic.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models_orm import ModelVersion


def register_from_artifacts(artifacts_dir: str | None = None) -> str:
    settings = get_settings()
    art = Path(artifacts_dir or settings.model_artifacts_dir)
    mpath = art / "metrics.json"
    if not mpath.exists():
        return f"no metrics.json in {art} -- nothing to register"

    m = json.loads(mpath.read_text())
    trained_at = m.get("trained_at")
    trained_dt = (
        dt.datetime.fromisoformat(trained_at) if trained_at else dt.datetime.now(dt.timezone.utc)
    )

    init_db()
    db = SessionLocal()
    try:
        # idempotent: don't duplicate the same training run
        existing = db.execute(
            select(ModelVersion).where(ModelVersion.trained_at == trained_dt)
        ).scalars().first()
        if existing:
            return f"ModelVersion for {trained_at} already registered (id={existing.id})"

        mv = ModelVersion(
            trained_at=trained_dt,
            algorithm=m.get("algorithm", "LightGBM"),
            threshold=float(m.get("threshold", settings.decision_threshold)),
            metrics={
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "false_positive_rate": m.get("false_positive_rate", 0.0),
                "macro_f1": m.get("macro_f1"),
            },
        )
        db.add(mv)
        db.commit()
        db.refresh(mv)
        return (f"registered ModelVersion id={mv.id} algo={mv.algorithm} "
                f"threshold={mv.threshold} metrics={mv.metrics}")
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default=None)
    args = ap.parse_args()
    print(register_from_artifacts(args.artifacts_dir))