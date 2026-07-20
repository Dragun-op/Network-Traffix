"""
predict_service.py -- one-shot scoring of a pasted CICIDS row for the
POST /api/predict endpoint (the "explainable predict" demo).

Reuses the exact same scorer + explainer the replay worker uses, so a manual
prediction goes through the identical path as a live-detected incident. Does NOT
write to the database -- this is a stateless "what would you say about this flow"
query, not an incident.
"""
from __future__ import annotations

from functools import lru_cache

from app.inference.explain import explain
from app.inference.model import NORMAL_LABELS, get_scorer
from app.inference.row_parser import parse_row


@lru_cache(maxsize=1)
def _scorer():
    # Loaded once and cached -- model.pkl + shap_explainer.pkl are ~30MB, so we
    # never reload them per request.
    return get_scorer()


def predict(row: str) -> dict:
    flow, ground_truth = parse_row(row)          # raises ValueError on bad input
    scorer = _scorer()
    scored = scorer.score(flow)
    contributions = explain(scorer, scored, flow)

    is_attack = scored.attack_type is not None
    return {
        "is_attack": is_attack,
        "attack_type": scored.attack_type,       # None => normal traffic
        "confidence": scored.confidence,
        "severity": scored.severity,
        "explanation": contributions,            # [{feature, contribution}], sums to 100
        "ground_truth": ground_truth,            # from the pasted row's Label, if present
        "model": scorer.name,                    # "trained_model" or "rule_based_v0"
    }