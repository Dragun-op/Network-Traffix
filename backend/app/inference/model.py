"""
Inference layer, hidden behind one function: `get_scorer()`.

Why this exists: the backend must not be blocked on the ML teammate's
training pipeline. `RuleBasedScorer` gives every downstream piece (API,
replay worker, tests) something real to call today. The moment
`model_artifacts_dir/model.pkl` + `features.json` exist, `get_scorer()` picks
up `TrainedModelScorer` instead -- nothing else in the codebase changes.

Both scorers implement the same contract:
    score(flow: dict) -> ScoredFlow(attack_type, confidence, severity, raw_features)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings

# Display names shown in the UI. The trained model is trained to emit these exact
# strings as its class labels (see ml/train.py DISPLAY_LABEL), and "BENIGN" is the
# sentinel that means "normal traffic -> raise no incident".
ATTACK_TYPES = [
    "DDoS", "DoS", "Port Scan", "Brute Force", "Botnet", "Web Attack", "Infiltration",
]
NORMAL_LABELS = ("Normal", "BENIGN")


@dataclass
class ScoredFlow:
    attack_type: str | None      # None means "classified as normal traffic"
    confidence: int              # 0-100
    severity: str                # low | medium | high | critical
    raw_features: dict = field(default_factory=dict)


def severity_from_confidence(confidence: int) -> str:
    if confidence >= 90:
        return "critical"
    if confidence >= 75:
        return "high"
    if confidence >= 50:
        return "medium"
    return "low"


class RuleBasedScorer:
    """
    Deterministic heuristic scorer. Not meant to be a real detector -- it
    exists so the API/replay/tests have a working, *explainable* baseline
    while the real model is being trained. Swapped out with zero interface
    changes once TrainedModelScorer is active.
    """

    name = "rule_based_v0"

    THRESHOLDS = {
        "syn_flag_count": 200,
        "flow_duration_ms": 50,
        "packets_per_second": 5000,
    }

    def score(self, flow: dict) -> ScoredFlow:
        syn = flow.get("syn_flag_count", 0)
        duration = flow.get("flow_duration_ms", 1000)
        pps = flow.get("packets_per_second", 0)

        signals = {
            "SYN Flag Count": min(syn / self.THRESHOLDS["syn_flag_count"], 1.0),
            "Flow Duration": min(self.THRESHOLDS["flow_duration_ms"] / max(duration, 1), 1.0),
            "Packets/s": min(pps / self.THRESHOLDS["packets_per_second"], 1.0),
        }
        score = sum(signals.values()) / len(signals)
        confidence = round(score * 100)

        if confidence < 30:
            return ScoredFlow(attack_type=None, confidence=confidence, severity="low")

        attack_type = flow.get("suggested_attack_type") or (
            "Port Scan" if syn > self.THRESHOLDS["syn_flag_count"] else "DDoS"
        )
        return ScoredFlow(
            attack_type=attack_type,
            confidence=confidence,
            severity=severity_from_confidence(confidence),
            raw_features=signals,
        )


class TrainedModelScorer:
    """
    Real scorer, activated automatically once the ML teammate's artifacts exist.
    Loads model.pkl + features.json once at process startup.

    Feature handling goes through the SAME feature_pipeline the model was trained
    with (via FeatureExtractor + preprocessor.json), so a live flow is turned into
    exactly the vector training used -- no train/serve skew. If preprocessor.json
    is absent it falls back to a raw positional lookup so the backend still runs.
    """

    name = "trained_model"

    def __init__(self, artifacts_dir: Path, threshold: float):
        import joblib  # imported lazily -- only required once artifacts exist

        self.model = joblib.load(artifacts_dir / "model.pkl")
        self.features: list[str] = json.loads((artifacts_dir / "features.json").read_text())
        self.classes = list(self.model.classes_)

        # Operating threshold: prefer the value train.py calibrated (threshold.json);
        # fall back to the backend's configured threshold otherwise.
        self.threshold = threshold
        tpath = artifacts_dir / "threshold.json"
        if tpath.exists():
            try:
                self.threshold = float(json.loads(tpath.read_text())["threshold"])
            except Exception:
                pass

        # Parity transform (same code as training). Optional so the backend never
        # hard-fails if only the raw model is present.
        self.extractor = None
        if (artifacts_dir / "preprocessor.json").exists():
            try:
                from app.inference.feature_extract import FeatureExtractor
                self.extractor = FeatureExtractor(str(artifacts_dir))
            except Exception:
                self.extractor = None

        shap_path = artifacts_dir / "shap_explainer.pkl"
        self.shap_explainer = joblib.load(shap_path) if shap_path.exists() else None

    def _vectorize(self, flow: dict):
        if self.extractor is not None:
            return self.extractor.transform_one(flow)        # (1, n) float32, parity
        # Fallback: raw positional lookup (missing -> 0). Used only if no
        # preprocessor.json shipped; no imputation/inf handling in this path.
        return [[flow.get(f, 0) for f in self.features]]

    def score(self, flow: dict) -> ScoredFlow:
        x = self._vectorize(flow)
        proba = self.model.predict_proba(x)[0]
        best_idx = max(range(len(proba)), key=lambda i: proba[i])
        label = self.classes[best_idx]
        confidence = round(float(proba[best_idx]) * 100)

        # Probability that this flow is ANY attack (1 - P(benign)); this is what
        # the calibrated threshold gates on.
        attack_proba = 1.0
        for i, c in enumerate(self.classes):
            if c in NORMAL_LABELS:
                attack_proba = 1.0 - float(proba[i])
                break

        if label in NORMAL_LABELS or attack_proba < self.threshold:
            return ScoredFlow(attack_type=None, confidence=confidence, severity="low")

        return ScoredFlow(
            attack_type=label,
            confidence=confidence,
            severity=severity_from_confidence(confidence),
        )


def get_scorer():
    settings = get_settings()
    artifacts_dir = Path(settings.model_artifacts_dir)
    required = ["model.pkl", "features.json"]
    if artifacts_dir.exists() and all((artifacts_dir / f).exists() for f in required):
        return TrainedModelScorer(artifacts_dir, settings.decision_threshold)
    return RuleBasedScorer()