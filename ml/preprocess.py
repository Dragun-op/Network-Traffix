"""
preprocess.py -- turn the 8 raw CICIDS-2017 CSVs into clean, split, model-ready
data plus the persisted preprocessing parameters the backend will reuse.

What it does, in order
----------------------
1. Load every CSV (memory-consciously: downcast numerics to float32 on load).
2. Concatenate -- ESSENTIAL, because each attack type lives in a different file,
   so a per-file split would put whole classes in only one split.
3. Build the coarse label, drop rows whose label is unknown/unmapped.
4. Stratified train / val / test split (70 / 15 / 15) on the coarse label, so
   even tiny classes (Infiltration ~36 rows) appear in every split.
5. Fit preprocessing params on the TRAIN split ONLY (drop-list + medians), then
   transform all three splits with those SAME params.
6. Save:
      artifacts/preprocessor.json   -- full transform spec (drops + medians + map)
      artifacts/features.json       -- just the ordered feature list (gap item A1)
      artifacts/label_distribution.json -- counts for the report/slides
      data/{train,val,test}.parquet -- features + coarse + raw label

Run:
    python preprocess.py --data-dir /path/to/csvs --out-dir .
    # options:
    #   --keep-dest-port     keep 'Destination Port' (default: drop it)
    #   --max-benign N       cap Benign rows to N (handles the 2.27M-row majority;
    #                        default: keep all)
    #   --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import feature_pipeline as fp


def _read_one(path: str) -> pd.DataFrame:
    """Read a single CICIDS CSV, downcasting numeric columns to float32 to keep
    the concatenated frame inside a small RAM budget. Label stays as category."""
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df = fp.clean_columns(df)
    for c in df.columns:
        if c == fp.LABEL_COL:
            continue
        # errors="coerce" turns any stray non-numeric token into NaN (handled later)
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
    df[fp.LABEL_COL] = df[fp.LABEL_COL].astype("category")
    return df


def load_all(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs found in {data_dir}")
    frames = []
    for f in files:
        t0 = time.time()
        d = _read_one(f)
        frames.append(d)
        print(f"  loaded {os.path.basename(f):45s} {len(d):>9,} rows "
              f"({time.time()-t0:4.1f}s)")
    df = pd.concat(frames, ignore_index=True)
    del frames
    print(f"  concatenated total: {len(df):,} rows x {df.shape[1]} cols")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--keep-dest-port", action="store_true",
                    help="keep 'Destination Port' instead of dropping it")
    ap.add_argument("--max-benign", type=int, default=None,
                    help="cap Benign rows to this many (default: keep all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    art_dir = os.path.join(args.out_dir, "artifacts")
    data_dir_out = os.path.join(args.out_dir, "data")
    os.makedirs(art_dir, exist_ok=True)
    os.makedirs(data_dir_out, exist_ok=True)

    print("[1/6] loading CSVs ...")
    df = load_all(args.data_dir)

    print("[2/6] building coarse label ...")
    # Keep the coarse label as a SEPARATE series -- never a column on df -- so the
    # feature pipeline can never mistake it for an input feature.
    y_all = fp.group_labels(df)
    unmapped = int(y_all.isna().sum())
    if unmapped:
        bad = df.loc[y_all.isna(), fp.LABEL_COL].astype(str).unique()[:10]
        print(f"  WARNING: {unmapped:,} rows had unmapped labels -> dropped. "
              f"examples: {list(bad)}")
        keep_mask = y_all.notna().to_numpy()
        df = df[keep_mask].reset_index(drop=True)
        y_all = y_all[keep_mask].reset_index(drop=True)

    # --- SCHEMA on the FULL dataset, BEFORE any capping ---------------------
    # Structural drops (constant + exact-duplicate columns) are computed here, on
    # every row, so the drop-list is deterministic: a column is dropped only if it
    # is constant or byte-identical to another across the ENTIRE dataset -- never
    # by a floating-point coincidence inside one sampled/capped split. Medians are
    # filled in later from TRAIN only.
    print("[3/6] computing schema on full data (deterministic drops) ...")
    schema = fp.fit_schema(df, drop_destination_port=not args.keep_dest_port)
    feature_order = schema["feature_order"]
    print(f"    kept {schema['n_features']} features; "
          f"dropped {len(schema['dropped_constant'])} constant, "
          f"{len(schema['dropped_duplicate'])} duplicate, "
          f"dest_port_dropped={schema['dropped_destination_port']}")
    print("    duplicate columns removed (dropped -> mirrors):")
    for dup, orig in sorted(schema["duplicate_of"].items()):
        print(f"        {dup:28s} == {orig}")

    # Optional: cap the Benign majority class (2.27M rows) for a faster, more
    # balanced demo model. Applied before splitting so split proportions stay honest.
    if args.max_benign is not None:
        ben_idx = y_all.index[y_all == "Benign"].to_numpy()
        if len(ben_idx) > args.max_benign:
            rng = np.random.RandomState(args.seed)
            drop_ben = rng.choice(ben_idx, size=len(ben_idx) - args.max_benign,
                                  replace=False)
            keep = np.ones(len(df), dtype=bool)
            keep[drop_ben] = False
            df = df[keep].reset_index(drop=True)
            y_all = y_all[keep].reset_index(drop=True)
            print(f"  capped Benign to {args.max_benign:,}; total now {len(df):,}")

    print("[4/6] class distribution (coarse):")
    dist = y_all.value_counts()
    for k, v in dist.items():
        print(f"    {k:14s} {v:>9,}")

    print("[5/6] stratified split 70/15/15 + medians on TRAIN ...")
    y = y_all.astype(str)
    idx = np.arange(len(df))
    # first carve off 30% (val+test), then halve it -> 15/15
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=args.seed, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=args.seed,
                              stratify=y.iloc[tmp])
    print(f"    train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    train_df, val_df, test_df = df.iloc[tr], df.iloc[va], df.iloc[te]
    y_tr, y_va, y_te = y_all.iloc[tr], y_all.iloc[va], y_all.iloc[te]

    # Learned parameter (imputation medians) comes from TRAIN only; merge onto the
    # full-data schema to form the final, persisted preprocessing spec.
    params = dict(schema)
    params["medians"] = fp.compute_medians(train_df, feature_order)
    params["median_source"] = "train"

    def build(split_df: pd.DataFrame, y_series: pd.Series) -> pd.DataFrame:
        X = fp.transform(split_df, params)                     # <-- same fn the backend calls
        out = pd.DataFrame(X, columns=params["feature_order"])
        out["label"] = y_series.to_numpy()
        out["label_raw"] = split_df[fp.LABEL_COL].astype(str).to_numpy()
        return out

    print("[6/6] transforming splits + saving ...")
    fp.save_params(params, os.path.join(art_dir, "preprocessor.json"))
    with open(os.path.join(art_dir, "features.json"), "w") as fh:
        json.dump(params["feature_order"], fh, indent=2)
    with open(os.path.join(art_dir, "label_distribution.json"), "w") as fh:
        json.dump({k: int(v) for k, v in dist.items()}, fh, indent=2)

    for name, sdf, ys in [("train", train_df, y_tr), ("val", val_df, y_va),
                          ("test", test_df, y_te)]:
        out = build(sdf, ys)
        path = os.path.join(data_dir_out, f"{name}.parquet")
        out.to_parquet(path, index=False)
        print(f"    wrote {path}  ({len(out):,} rows, {out.shape[1]} cols)")

    print("\nDONE. Artifacts in", art_dir)


if __name__ == "__main__":
    main()
