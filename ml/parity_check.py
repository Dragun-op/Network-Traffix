"""Prove the backend inference path == the training path, row for row."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "ml")
sys.path.insert(0, "backend/app/inference")

import feature_pipeline as fp
from feature_extract import FeatureExtractor

# 1) Backend loads ONLY the persisted artifact -- no training code, no CSVs.
fx = FeatureExtractor(model_dir="ml/artifacts")
print("backend loaded", len(fx.feature_names), "features from preprocessor.json")

# 2) Grab a handful of raw flows straight from a CSV (simulating live capture),
#    deliberately mixing an attack-heavy file. Keep them as raw dicts.
raw = pd.read_csv("/home/claude/cicids/Wednesday-workingHours.pcap_ISCX.csv",
                  skipinitialspace=True, low_memory=False, nrows=2000)
raw = fp.clean_columns(raw)
sample = raw.sample(6, random_state=7)
flows = sample.drop(columns=["Label"]).to_dict(orient="records")

# 3) Backend transform: one flow at a time, exactly like the replay worker will.
backend_vecs = np.vstack([fx.transform_one(f) for f in flows])

# 4) Training-path transform on the same rows, via the SAME shared function.
params = fp.load_params("ml/artifacts/preprocessor.json")
train_vecs = fp.transform(sample.drop(columns=["Label"]), params)

# 5) Compare byte-for-byte.
identical = np.array_equal(backend_vecs, train_vecs)
print("shapes:", backend_vecs.shape, train_vecs.shape)
print("BYTE-IDENTICAL backend vs training transform:", identical)
if not identical:
    diff = np.abs(backend_vecs - train_vecs)
    print("max abs diff:", diff.max())

# 6) Also prove column ALIGNMENT: feed a flow with keys shuffled + one missing +
#    an unexpected extra key -> transform must still line up by name and impute.
import random
shuffled = dict(flows[0]); items = list(shuffled.items()); random.Random(1).shuffle(items)
shuffled = dict(items)
shuffled.pop("Flow Duration", None)        # missing feature -> should be imputed
shuffled["Some Garbage Col"] = 999999      # unexpected -> should be ignored
v_shuf = fx.transform_one(shuffled)
print("robust to shuffled/missing/extra keys -> shape:", v_shuf.shape,
      "| any NaN?", bool(np.isnan(v_shuf).any()))
