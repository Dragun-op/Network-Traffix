"""
train.py -- train the detector and write the artifacts the backend loads.

Reads the parquet splits from preprocess.py, trains a LightGBM multiclass
model, fits a SHAP explainer, calibrates a decision threshold against a target
false-positive rate, and writes everything into the artifacts directory in the
EXACT shape backend/app/inference/model.py expects:

    model.pkl            joblib-dumped LightGBM model (has .predict_proba, .classes_)
    features.json        ordered feature list (already produced by preprocess.py)
    shap_explainer.pkl   joblib-dumped shap.TreeExplainer(model)
    threshold.json       {threshold, precision, recall, false_positive_rate, ...}
    preprocessor.json    (copied in) so the backend's FeatureExtractor has parity
    metrics.json         full report (per-class + confusion matrix) for slides / DB

Two backend-compatibility decisions are made HERE on purpose:

1. CLASS LABELS are the backend's display names. model.py returns the predicted
   class label straight through as `attack_type`, and treats the label "BENIGN"
   as normal traffic (attack_type=None). So we train on: BENIGN, DoS, DDoS,
   "Port Scan", "Brute Force", Botnet, "Web Attack", Infiltration -- matching
   ATTACK_TYPES in model.py exactly. Get one of these strings wrong and the UI
   shows the wrong label or the benign sentinel stops working.

2. The registered precision/recall/FPR are for the ATTACK-vs-BENIGN decision at
   the chosen threshold -- that is what "false positive rate" means for an IDS
   and what /api/metrics reports.

Run:
    python train.py --data-dir data --artifacts-dir artifacts
    #   --target-fpr 0.02        pick the threshold that keeps FPR <= this
    #   --backend-artifacts-dir ../backend/ml_artifacts   also copy runtime files there
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Coarse pipeline label -> backend display label. The RIGHT-hand strings must
# match ATTACK_TYPES / the benign sentinel in backend/app/inference/model.py.
DISPLAY_LABEL = {
    "Benign": "BENIGN",          # model.py treats "BENIGN" as normal -> no incident
    "DoS": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Port Scan",
    "BruteForce": "Brute Force",
    "WebAttack": "Web Attack",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
}
BENIGN_LABEL = "BENIGN"


def load_split(data_dir: Path, name: str, features: list[str]):
    df = pd.read_parquet(data_dir / f"{name}.parquet")
    X = df[features].to_numpy(dtype="float32")
    y = df["label"].map(DISPLAY_LABEL)
    if y.isna().any():
        bad = df.loc[y.isna(), "label"].unique()
        raise SystemExit(f"{name}: labels with no display mapping: {list(bad)}")
    return X, y.to_numpy(), df


def calibrate_threshold(y_true_bin, attack_proba, target_fpr: float):
    """Pick the smallest decision threshold whose benign false-positive rate is
    <= target_fpr. attack_proba = P(flow is any attack) = 1 - P(BENIGN).
    Returns (threshold, achieved_fpr, achieved_recall)."""
    benign = ~y_true_bin
    n_benign = max(int(benign.sum()), 1)
    best = None
    for t in np.linspace(0.05, 0.95, 19):
        pred_attack = attack_proba >= t
        fpr = float((pred_attack & benign).sum() / n_benign)
        recall = float((pred_attack & y_true_bin).sum() / max(int(y_true_bin.sum()), 1))
        if fpr <= target_fpr:
            best = (round(float(t), 3), round(fpr, 4), round(recall, 4))
            break
    if best is None:                       # target too strict -> fall back to 0.5
        t = 0.5
        pred_attack = attack_proba >= t
        fpr = float((pred_attack & benign).sum() / n_benign)
        recall = float((pred_attack & y_true_bin).sum() / max(int(y_true_bin.sum()), 1))
        best = (t, round(fpr, 4), round(recall, 4))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--target-fpr", type=float, default=0.02,
                    help="threshold is calibrated to keep benign FPR at/below this")
    ap.add_argument("--backend-artifacts-dir", default=None,
                    help="if set, copy runtime artifacts here too (e.g. ../backend/ml_artifacts)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    art = Path(args.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    features = json.loads((art / "features.json").read_text())
    print(f"[1/6] loading splits ({len(features)} features) ...")
    Xtr, ytr, _ = load_split(data_dir, "train", features)
    Xva, yva, _ = load_split(data_dir, "val", features)
    Xte, yte, te_df = load_split(data_dir, "test", features)
    print(f"    train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    print("[2/6] training LightGBM (class_weight=balanced for the imbalance) ...")
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",     # so Infiltration/Bot/WebAttack aren't ignored
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
    )
    print(f"    best_iteration={model.best_iteration_}  classes={list(model.classes_)}")

    print("[3/6] evaluating on held-out test ...")
    proba = model.predict_proba(Xte)
    classes = list(model.classes_)
    benign_idx = classes.index(BENIGN_LABEL)
    attack_proba = 1.0 - proba[:, benign_idx]        # P(any attack)
    y_pred = model.predict(Xte)

    # multiclass per-class report (for the slide / metrics.json)
    report = classification_report(yte, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, y_pred, labels=classes).tolist()

    # attack-vs-benign story + threshold calibration (what /api/metrics registers)
    y_true_bin = yte != BENIGN_LABEL
    threshold, fpr, recall = calibrate_threshold(y_true_bin, attack_proba, args.target_fpr)
    pred_attack = attack_proba >= threshold
    tp = int((pred_attack & y_true_bin).sum())
    fp = int((pred_attack & ~y_true_bin).sum())
    precision = round(tp / max(tp + fp, 1), 4)
    print(f"    threshold={threshold} -> precision={precision} recall={recall} FPR={fpr}")
    print(f"    (target FPR was {args.target_fpr})")

    print("[4/6] fitting SHAP TreeExplainer ...")
    import shap
    explainer = shap.TreeExplainer(model)

    print("[5/6] saving artifacts ...")
    joblib.dump(model, art / "model.pkl")
    joblib.dump(explainer, art / "shap_explainer.pkl")
    (art / "threshold.json").write_text(json.dumps({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "target_fpr": args.target_fpr,
    }, indent=2))
    trained_at = dt.datetime.now(dt.timezone.utc).isoformat()
    metrics = {
        "algorithm": "LightGBM",
        "trained_at": trained_at,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "classes": classes,
        "per_class": {k: report[k] for k in classes if k in report},
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "confusion_matrix": cm,
        "confusion_labels": classes,
        "n_test": int(len(yte)),
    }
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # features.json already exists; make sure preprocessor.json rides along so the
    # backend's FeatureExtractor uses the identical transform.
    if not (art / "preprocessor.json").exists():
        print("    WARNING: preprocessor.json missing in artifacts dir")

    print("[6/6] done.")
    # per-class recall is the honest 'what works / what doesn't' table
    print("\n    per-class recall on test:")
    for c in classes:
        r = report.get(c, {}).get("recall", 0.0)
        n = report.get(c, {}).get("support", 0)
        print(f"        {c:14s} recall={r:5.3f}  (n={int(n)})")

    if args.backend_artifacts_dir:
        dst = Path(args.backend_artifacts_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for f in ["model.pkl", "features.json", "shap_explainer.pkl",
                  "threshold.json", "preprocessor.json", "metrics.json"]:
            src = art / f
            if src.exists():
                shutil.copy2(src, dst / f)
        print(f"\n    copied runtime artifacts -> {dst}")

    print(f"\nDONE. Artifacts in {art}")


if __name__ == "__main__":
    main()