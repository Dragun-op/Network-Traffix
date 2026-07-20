"""
feature_extract.py -- the backend's ONLY entry point for turning an incoming
flow into a model input.

It does NOT re-implement any feature logic. It imports the exact same
`feature_pipeline` module the trainer used and drives it with the parameters
that were persisted at training time (artifacts/preprocessor.json). That is what
makes serving byte-for-byte compatible with training: same drop-list, same
feature order, same imputation medians -- none of it recomputed here.

Typical use inside the replay worker / inference engine:

    fx = FeatureExtractor(model_dir="ml/artifacts")   # load once at startup
    row = {"Destination Port": 80, "Flow Duration": 12345, ...}  # one raw flow
    x = fx.transform_one(row)        # -> shape (1, n_features) float32
    proba = model.predict_proba(x)   # feed straight into the model

The single-source-of-truth import path:
    ml/feature_pipeline.py  <-- trainer AND this file both import it.
In the real repo, ship feature_pipeline.py as a small shared package (or copy it
next to this file) so both trees resolve the same module.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# Resolve the shared pipeline whether it's installed as a package or sitting
# alongside the ml/ code. Adjust the fallback path to your repo layout.
try:
    import feature_pipeline as fp                       # shared package on path
except ModuleNotFoundError:                              # pragma: no cover
    import importlib.util
    _here = os.path.dirname(__file__)
    _shared = os.path.join(_here, "..", "..", "..", "ml", "feature_pipeline.py")
    _spec = importlib.util.spec_from_file_location("feature_pipeline",
                                                   os.path.abspath(_shared))
    fp = importlib.util.module_from_spec(_spec)          # type: ignore
    _spec.loader.exec_module(fp)                         # type: ignore


class FeatureExtractor:
    """Loads the persisted preprocessing params once and applies them to live flows."""

    def __init__(self, model_dir: str = "ml/artifacts") -> None:
        self.params = fp.load_params(os.path.join(model_dir, "preprocessor.json"))
        self.feature_names: list[str] = self.params["feature_order"]

    def transform_one(self, flow: Mapping[str, Any]) -> np.ndarray:
        """One raw flow (dict of raw CICIDS column -> value) -> (1, n_features)."""
        df = pd.DataFrame([dict(flow)])
        return fp.transform(df, self.params)             # identical to training

    def transform_many(self, flows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Batch version for the replay worker (N raw flows -> (N, n_features))."""
        df = pd.DataFrame([dict(f) for f in flows])
        return fp.transform(df, self.params)
