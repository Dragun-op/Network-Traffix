"""
feature_pipeline.py -- the SINGLE source of truth for turning a raw CICIDS-2017
flow record into a model-ready feature vector.

Why this file exists
--------------------
Both sides of the system import this exact module:

    * offline training      -> ml/train.py
    * online inference       -> backend/app/inference/feature_extract.py

Because there is only ONE definition of "what a feature vector is", the features
the model was trained on can never drift away from the features the backend
serves. That was gap item A1 ("training and backend must not drift").

The contract
------------
    fit(train_df)            -> params      # LEARN drop-list, feature order, medians
    save_params(params, path)                # persist to preprocessor.json
    load_params(path)        -> params       # reload at serve time
    transform(df, params)    -> X (float32)  # works on a full frame OR one row
    group_labels(df, params) -> y            # coarse 7-attack + Benign taxonomy

Everything the transform depends on -- which columns to drop, the exact ordered
feature list, and the median value used to fill each missing cell -- is LEARNED
on the training split and PERSISTED to JSON. Inference reloads those exact
numbers, so the same raw flow always produces the same vector. Nothing about the
transform is recomputed at serve time.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

# The label column in the raw CICIDS CSVs (after we strip whitespace it is "Label").
LABEL_COL = "Label"

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------
# Raw CICIDS-2017 has 15 labels. Several are near-identical attack variants and a
# few are extremely rare in the full dataset (Heartbleed=11, Web Sql Injection=21,
# Infiltration=36 rows). We fold the variants into 7 attack families + Benign.
# Keeping this as an explicit dict makes the choice easy to defend to a judge and
# easy to change -- to go back to 15-class, just train on the normalized raw label
# instead of the grouped one.
LABEL_GROUPS = {
    "BENIGN": "Benign",
    # --- Denial of Service family (all volumetric/resource-exhaustion single-host) ---
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",          # 11 rows -- folded in rather than left as its own class
    # --- Distributed DoS (kept separate: multi-source, different signature) ---
    "DDoS": "DDoS",
    # --- Reconnaissance ---
    "PortScan": "PortScan",
    # --- Credential brute force ---
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    # --- Web application attacks (labels normalized first -- see normalize_label) ---
    "Web Attack Brute Force": "WebAttack",
    "Web Attack XSS": "WebAttack",
    "Web Attack Sql Injection": "WebAttack",
    # --- Botnet C2 ---
    "Bot": "Bot",
    # --- Infiltration (36 rows -- tiny; flagged for stratification) ---
    "Infiltration": "Infiltration",
}


def normalize_label(raw: str) -> str:
    """Repair the known mojibake in CICIDS-2017 web-attack labels.

    The raw files store 'Web Attack \x96 Brute Force', where 0x96 (an en dash)
    was mis-encoded and surfaces as the Unicode replacement char U+FFFD ('�').
    We turn any such separator into a plain space and collapse whitespace, so the
    label matches the keys in LABEL_GROUPS.
    """
    s = str(raw).replace("\x96", " ").replace("\ufffd", " ")
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Column hygiene
# ---------------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from column names (raw headers look like
    ' Flow Duration'). Pure rename -- drops nothing, copies no data."""
    return df.rename(columns=lambda c: str(c).strip())


def _constant_columns(df: pd.DataFrame) -> list[str]:
    """Columns with a single unique value (NaN included) -- they carry no signal.
    This catches the dead flag columns (Bwd PSH Flags, Fwd URG Flags, CWE Flag
    Count, the Bulk-rate group, ...)."""
    return [c for c in df.columns
            if c != LABEL_COL and df[c].nunique(dropna=False) == 1]


def _duplicate_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return {duplicate_col: kept_col} for columns byte-identical to an earlier
    one (we keep the first occurrence).

    Detected by CONTENT HASH + an exact equality check, not by name. This catches
    CICIDS-2017's 'Fwd Header Length', which appears twice in the raw header
    (pandas renames the 2nd copy to 'Fwd Header Length.1' and keeps both), AND the
    dataset's built-in feature aliases -- e.g. 'Avg Fwd Segment Size' is defined
    identically to 'Fwd Packet Length Mean', 'Subflow Fwd Packets' to
    'Total Fwd Packets'. nunique()==1 misses all of these, because a duplicated
    column is not constant -- just redundant.
    """
    seen: dict[int, str] = {}
    dupes: dict[str, str] = {}
    for col in df.columns:
        if col == LABEL_COL:
            continue
        h = int(pd.util.hash_pandas_object(df[col], index=False).sum())
        if h in seen and df[col].equals(df[seen[h]]):
            dupes[col] = seen[h]        # identical content -> redundant (keep seen[h])
        else:
            seen[h] = col
    return dupes


# ---------------------------------------------------------------------------
# fit / transform
# ---------------------------------------------------------------------------
def compute_medians(df: pd.DataFrame, feature_order: list[str]) -> dict[str, float]:
    """Per-feature median for imputation. Medians (not means) because the
    flow-rate features are heavy-tailed. Call on the TRAIN split to stay
    leakage-safe. inf is treated as missing before the median is taken."""
    md = clean_columns(df)
    feats = md.reindex(columns=feature_order).replace([np.inf, -np.inf], np.nan)
    med = feats.median(numeric_only=True).fillna(0.0)
    return {k: float(v) for k, v in med.items()}


def fit_schema(schema_df: pd.DataFrame, drop_destination_port: bool = True) -> dict:
    """Decide the STRUCTURAL transform (which columns to drop + the feature order).

    This is pure schema hygiene -- it uses no label information and cannot leak, so
    it is safe (and preferable) to run on the FULL dataset. Doing so makes the
    drop-list deterministic: a column is dropped only if it is constant, or
    byte-identical to another column across EVERY row -- never by a floating-point
    coincidence inside one sampled/capped split. Does NOT compute medians (that is
    a learned statistic; see compute_medians), which also keeps this step light
    enough to run on millions of rows.

    drop_destination_port : In the CICIDS testbed, attacks were launched at fixed
        ports, so 'Destination Port' can leak the label. Default True for a cleaner
        generalization story; set False to keep it as a legitimate service signal
        (80=HTTP, 443=HTTPS, ...) and be ready to defend the choice.
    """
    df = clean_columns(schema_df)

    drop: set[str] = set()
    if drop_destination_port and "Destination Port" in df.columns:
        drop.add("Destination Port")

    # Find CONSTANT columns first and set them aside, then look for duplicates ONLY
    # among the columns that actually vary. Otherwise every all-zero column would be
    # reported as a "duplicate" of the first all-zero column, which is misleading --
    # those are constant, not meaningfully redundant.
    const = _constant_columns(df)
    non_const = [c for c in df.columns if c not in const]
    dup_map = _duplicate_columns(df[non_const])   # {dup_col: kept_col}, varying cols only
    dup = list(dup_map)
    drop.update(const)
    drop.update(dup)

    feature_order = [c for c in df.columns if c not in drop and c != LABEL_COL]

    return {
        "feature_order": feature_order,
        "n_features": len(feature_order),
        "dropped_constant": sorted(const),
        "dropped_duplicate": sorted(dup),
        "duplicate_of": dup_map,          # each dropped duplicate -> the column it mirrors
        "dropped_destination_port": bool(drop_destination_port
                                         and "Destination Port" in df.columns),
        "label_groups": LABEL_GROUPS,
    }


def fit(schema_df: pd.DataFrame,
        drop_destination_port: bool = True,
        median_df: pd.DataFrame | None = None) -> dict:
    """Convenience: full fit = structural schema + imputation medians, in one call.

    Preferred usage for a proper train/val/test pipeline is to call `fit_schema`
    on the FULL data and `compute_medians` on the TRAIN split separately (see
    preprocess.py), so structural drops are deterministic and medians stay
    leakage-safe. This wrapper is handy when you just want to fit everything on a
    single frame (e.g. quick experiments): pass `median_df` to source medians from
    the train split, or leave it None to use `schema_df` for both.
    """
    params = fit_schema(schema_df, drop_destination_port=drop_destination_port)
    src = median_df if median_df is not None else schema_df
    params["medians"] = compute_medians(src, params["feature_order"])
    params["median_source"] = "train" if median_df is not None else "schema_df"
    return params


def transform(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Apply the LEARNED transform. Deterministic given `params`.

    Works on a full DataFrame or a single-row DataFrame -- the backend builds a
    one-row frame from an incoming flow dict and calls this exact function, which
    is what guarantees train/serve parity.

    Steps: clean names -> reindex to the exact training feature set (missing
    columns become NaN, unexpected columns are dropped) -> inf->NaN -> fill each
    column with its learned median -> return a float32 matrix in feature order.
    """
    df = clean_columns(df)
    order = params["feature_order"]
    med = params["medians"]

    X = df.reindex(columns=order)                       # enforce exact columns + order
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(value={c: med.get(c, 0.0) for c in order})
    return X.astype("float32").to_numpy()


def group_labels(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Map raw labels to the coarse taxonomy. Unknown labels -> NaN (so they are
    easy to spot rather than silently mislabeled)."""
    groups = (params or {}).get("label_groups", LABEL_GROUPS)
    df = clean_columns(df)
    return df[LABEL_COL].map(normalize_label).map(groups)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def save_params(params: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(params, fh, indent=2)


def load_params(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)
