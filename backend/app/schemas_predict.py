"""
Request/response shapes for POST /api/predict. Kept in their own file so the
existing schemas.py stays untouched.
"""
from __future__ import annotations

from pydantic import BaseModel

from app.schemas import FeatureContribution


class PredictIn(BaseModel):
    row: str          # a raw CICIDS-2017 CSV line (comma-separated), with or without Label


class PredictOut(BaseModel):
    is_attack: bool
    attack_type: str | None
    confidence: int
    severity: str
    explanation: list[FeatureContribution]
    ground_truth: str | None = None
    model: str