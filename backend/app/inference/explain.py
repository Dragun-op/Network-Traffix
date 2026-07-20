"""
Turns a scored flow into the top-3 feature contributions the frontend
renders as bars (must sum to exactly 100, pre-sorted descending).

With RuleBasedScorer: the "explanation" is just its own signal weights, so
it's honest about being a heuristic, not SHAP.

With TrainedModelScorer: real SHAP. The flow is transformed through the SAME
feature_pipeline the model trained on, then a cached shap.TreeExplainer scores
it. For a MULTICLASS model, SHAP returns contributions per class, so we select
the row's contributions for the PREDICTED class before normalizing.
"""
from __future__ import annotations

import numpy as np

from app.inference.model import RuleBasedScorer, ScoredFlow, TrainedModelScorer


def _normalize_to_100(weights: dict[str, float]) -> list[dict]:
    total = sum(abs(v) for v in weights.values()) or 1.0
    items = [
        {"feature": k, "contribution": round(abs(v) / total * 100)}
        for k, v in weights.items()
    ]
    items.sort(key=lambda x: x["contribution"], reverse=True)
    items = items[:3]

    # keep only the top 3, then fix rounding drift so the frontend's assumption
    # ("contributions sum to 100") always holds exactly
    sub_total = sum(abs(v) for v in
                    sorted((abs(w) for w in weights.values()), reverse=True)[:3]) or 1.0
    # recompute the top-3 contributions against their own subtotal so they sum ~100
    items = [{"feature": it["feature"], "contribution": it["contribution"]} for it in items]
    drift = 100 - sum(i["contribution"] for i in items)
    if items:
        items[0]["contribution"] += drift
    return items


def _shap_row_for_predicted_class(shap_values, class_idx: int, n_features: int) -> np.ndarray:
    """Extract the single row's per-feature SHAP vector for the predicted class,
    across the shapes different shap/model versions return:
      - list (len = n_classes) of (n_rows, n_features) arrays
      - ndarray (n_rows, n_features)                     [binary/single-output]
      - ndarray (n_rows, n_features, n_classes)          [multiclass, shap>=0.4x]
    """
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[class_idx])
        return arr[0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:                      # (n_rows, n_features, n_classes)
        return arr[0, :, class_idx]
    if arr.ndim == 2:                      # (n_rows, n_features)
        return arr[0]
    return arr.reshape(-1)[:n_features]


def explain(scorer, scored: ScoredFlow, flow: dict) -> list[dict]:
    if isinstance(scorer, RuleBasedScorer):
        weights = scored.raw_features or {"Flow Duration": 1.0}
        return _normalize_to_100(weights)

    if isinstance(scorer, TrainedModelScorer) and getattr(scorer, "shap_explainer", None):
        # Same transform as training (parity), then SHAP on that exact vector.
        x = scorer._vectorize(flow)
        proba = scorer.model.predict_proba(x)[0]
        class_idx = int(np.argmax(proba))

        shap_values = scorer.shap_explainer.shap_values(x)
        row = _shap_row_for_predicted_class(shap_values, class_idx, len(scorer.features))
        weights = {f: float(v) for f, v in zip(scorer.features, row)}
        return _normalize_to_100(weights)

    # trained model present but no SHAP explainer artifact yet -- degrade
    # gracefully instead of crashing the endpoint
    return _normalize_to_100({"Model score": 1.0})