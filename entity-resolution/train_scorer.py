#!/usr/bin/env python3
"""
Train a logistic regression scorer for gray zone pairs.

Combines demographic comparison features with LLM confidence to classify
gray zone pairs as match/non-match. Outputs a trained model and metrics.

Produces:
  scorer_model.joblib    — trained LogisticRegression + StandardScaler
  scorer_metrics.json    — cross-validated precision, recall, F1
"""

import json
import logging
import math
import sys
from itertools import combinations
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Add paths for imports inside Docker container (/app)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_script_dir))    # entity-resolution/ -> import src.*
sys.path.insert(0, str(_project_root))  # project root -> import shared.*

from src.data_loader import create_record_id  # noqa: E402
from shared.data_loader import load_facility_patients  # noqa: E402
from shared.ground_truth import load_ground_truth, add_record_ids_to_ground_truth  # noqa: E402

logger = logging.getLogger(__name__)

DEMO_FEATURES = [
    "first_name_sim", "last_name_sim", "address_sim", "city_sim",
    "state_match", "zip_match", "ssn_match", "birthdate_match",
    "total_score", "name_score", "address_score",
]
ALL_FEATURES = DEMO_FEATURES + ["llm_logit"]


def build_dataset(
    features_path: str,
    predictions_path: str,
    augmented_dir: str,
) -> pd.DataFrame:
    """Build labeled dataset for gray zone pairs.

    Joins demographic features with LLM predictions and ground truth labels.
    Returns a DataFrame with feature columns + 'label' (0/1).
    """
    # Load ground truth
    run_dirs = sorted(Path(augmented_dir).glob("run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directory found in {augmented_dir}")
    run_dir = run_dirs[-1]

    patients_df = load_facility_patients(str(run_dir))
    patients_df = create_record_id(patients_df)
    gt_df = load_ground_truth(str(run_dir))
    gt_df = add_record_ids_to_ground_truth(gt_df, patients_df)

    true_pairs = set()
    for _, group in gt_df.groupby("true_patient_id"):
        rids = sorted(group["record_id"].tolist())
        for a, b in combinations(rids, 2):
            true_pairs.add((a, b))

    # Load features
    feat_df = pd.read_csv(features_path)
    feat_lookup = {}
    for _, row in feat_df.iterrows():
        pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
        feat_lookup[pair] = row

    # Load predictions and join with features
    pred_df = pd.read_csv(predictions_path)

    rows = []
    skipped = 0
    for _, row in pred_df.iterrows():
        pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
        feat_row = feat_lookup.get(pair)
        if feat_row is None:
            skipped += 1
            continue

        pred = int(row["prediction"])
        conf = row["confidence"]
        match_prob = conf if pred == 1 else (1.0 - conf)
        match_prob = max(1e-4, min(1 - 1e-4, match_prob))
        llm_logit = math.log(match_prob / (1.0 - match_prob))

        r = {
            "label": 1 if pair in true_pairs else 0,
            "llm_logit": llm_logit,
        }
        for col in DEMO_FEATURES:
            r[col] = feat_row[col]
        rows.append(r)

    if skipped:
        logger.warning(f"Skipped {skipped} predictions not found in features")

    return pd.DataFrame(rows)


@click.command()
@click.option("--augmented-dir", required=True, type=click.Path(exists=True),
              help="Path to augmented data directory (for ground truth)")
@click.option("--features", required=True, type=click.Path(exists=True),
              help="Path to features.csv from resolve stage")
@click.option("--predictions", required=True, type=click.Path(exists=True),
              help="Path to predictions.csv from infer stage")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Output directory for model and metrics")
@click.option("--cv-folds", default=5, type=int,
              help="Number of cross-validation folds")
def main(augmented_dir, features, predictions, output_dir, cv_folds):
    """Train a logistic regression scorer for gray zone entity resolution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Build dataset ---
    logger.info("Building labeled dataset...")
    df = build_dataset(features, predictions, augmented_dir)
    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    logger.info(f"Dataset: {len(df)} samples ({n_pos} positive, {n_neg} negative)")

    if n_pos < 5:
        logger.error(f"Too few positive samples ({n_pos}) to train a model")
        sys.exit(1)

    X = df[ALL_FEATURES].values
    y = df["label"].values

    # --- Cross-validated evaluation ---
    logger.info(f"Running {cv_folds}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        )
        model.fit(X_tr, y[train_idx])
        oof_preds[test_idx] = model.predict(X_te)

        tp = int(((oof_preds[test_idx] == 1) & (y[test_idx] == 1)).sum())
        fp = int(((oof_preds[test_idx] == 1) & (y[test_idx] == 0)).sum())
        fn = int(((oof_preds[test_idx] == 0) & (y[test_idx] == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        fold_metrics.append({"fold": fold + 1, "tp": tp, "fp": fp, "fn": fn,
                             "precision": p, "recall": r, "f1": f1})
        logger.info(f"  Fold {fold+1}: P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # Overall CV metrics
    tp = int(((oof_preds == 1) & (y == 1)).sum())
    fp = int(((oof_preds == 1) & (y == 0)).sum())
    fn = int(((oof_preds == 0) & (y == 1)).sum())
    cv_p = tp / (tp + fp) if (tp + fp) else 0
    cv_r = tp / (tp + fn) if (tp + fn) else 0
    cv_f1 = 2 * cv_p * cv_r / (cv_p + cv_r) if (cv_p + cv_r) else 0
    logger.info(f"  CV overall: P={cv_p:.3f} R={cv_r:.3f} F1={cv_f1:.3f}")

    # --- Train final model on all data ---
    logger.info("Training final model on all data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_scaled, y)

    # Log coefficients
    for feat, coef in zip(ALL_FEATURES, model.coef_[0]):
        logger.info(f"  {feat}: {coef:.4f}")
    logger.info(f"  intercept: {model.intercept_[0]:.4f}")

    # --- Save model ---
    model_data = {
        "model": model,
        "scaler": scaler,
        "features": ALL_FEATURES,
    }
    model_path = out / "scorer_model.joblib"
    joblib.dump(model_data, model_path)
    logger.info(f"Saved model to {model_path}")

    # --- Save metrics ---
    metrics = {
        "dataset_size": len(df),
        "positive_count": n_pos,
        "negative_count": n_neg,
        "features": ALL_FEATURES,
        "cv_folds": cv_folds,
        "cv_precision": cv_p,
        "cv_recall": cv_r,
        "cv_f1": cv_f1,
        "cv_tp": tp,
        "cv_fp": fp,
        "cv_fn": fn,
        "fold_metrics": fold_metrics,
        "coefficients": {feat: float(coef) for feat, coef in zip(ALL_FEATURES, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
    }

    metrics_path = out / "scorer_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    logger.info("Train scorer complete!")


if __name__ == "__main__":
    main()
