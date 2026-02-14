#!/usr/bin/env python3
"""
Golden Records stage: combine auto-matches + LLM predictions -> golden records -> evaluation.

Produces:
  golden_records.csv       — master patient index with facility provenance
  all_matches.csv          — all matched pairs with source (auto_match/llm) and scores
  evaluation_metrics.json  — precision, recall, F1
"""

import json
import logging
import math
import sys
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
import yaml

# Add paths for imports inside Docker container (/app)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_script_dir))    # entity-resolution/ -> import src.*
sys.path.insert(0, str(_project_root))  # project root -> import shared.*

from src.data_loader import create_record_id  # noqa: E402
from src.golden_record import create_golden_records  # noqa: E402
from src.evaluation import evaluate_matches, evaluate_golden_records  # noqa: E402
from shared.data_loader import load_facility_patients  # noqa: E402
from shared.ground_truth import load_ground_truth, add_record_ids_to_ground_truth  # noqa: E402

logger = logging.getLogger(__name__)


@click.command()
@click.option("--augmented-dir", required=True, type=click.Path(exists=True),
              help="Path to augmented data directory")
@click.option("--auto-matches", required=True, type=click.Path(exists=True),
              help="Path to auto_matches.csv from resolve stage")
@click.option("--predictions", required=True, type=click.Path(exists=True),
              help="Path to predictions.csv from infer stage")
@click.option("--gray-zone-pairs", required=False, default=None, type=click.Path(exists=True),
              help="Path to gray_zone_pairs.csv (only needed if predictions.csv lacks record_id columns)")
@click.option("--features", "features_path", required=False, default=None, type=click.Path(exists=True),
              help="Path to features.csv from resolve stage")
@click.option("--scorer-model", "scorer_model_path", required=False, default=None, type=click.Path(exists=True),
              help="Path to scorer_model.joblib from train_scorer stage")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Output directory for golden records")
@click.option("--config", required=True, type=click.Path(exists=True),
              help="Path to matching config YAML")
def main(augmented_dir, auto_matches, predictions, gray_zone_pairs, features_path, scorer_model_path, output_dir, config):
    """Combine auto-matches + LLM predictions, build golden records, evaluate."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(config) as f:
        cfg = yaml.safe_load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Discover run directory
    run_dirs = sorted(Path(augmented_dir).glob("run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directory found in {augmented_dir}")
    run_dir = run_dirs[-1]
    logger.info(f"Using run directory: {run_dir}")

    # Load patient data
    patients_df = load_facility_patients(str(run_dir))
    patients_df = create_record_id(patients_df)
    logger.info(f"Loaded {len(patients_df)} patient records")

    # Load ground truth
    ground_truth_df = load_ground_truth(str(run_dir))
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    # --- Step 1: Load auto-matches ---
    auto_df = pd.read_csv(auto_matches)
    auto_pairs = set()
    auto_scores = {}
    for _, row in auto_df.iterrows():
        pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
        auto_pairs.add(pair)
        auto_scores[pair] = row["total_score"]
    logger.info(f"Auto-matches: {len(auto_pairs)}")

    # --- Step 2: Score gray zone predictions ---
    pred_df = pd.read_csv(predictions)

    # Load features for demographic scores
    feat_lookup = {}
    if features_path is not None:
        feat_df = pd.read_csv(features_path)
        for _, row in feat_df.iterrows():
            pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
            feat_lookup[pair] = row
        logger.info(f"Loaded {len(feat_lookup)} feature vectors from {features_path}")

    # Parse predictions into per-pair records
    pred_records = []
    if "record_id_1" in pred_df.columns and "record_id_2" in pred_df.columns:
        for _, row in pred_df.iterrows():
            pred_records.append({
                "pair": tuple(sorted([row["record_id_1"], row["record_id_2"]])),
                "prediction": int(row["prediction"]),
                "confidence": row.get("confidence"),
            })
    else:
        if gray_zone_pairs is None:
            raise click.UsageError(
                "--gray-zone-pairs required when predictions.csv lacks record_id columns"
            )
        gz_df = pd.read_csv(gray_zone_pairs)
        if len(gz_df) != len(pred_df):
            logger.warning(
                f"Row count mismatch: gray_zone_pairs={len(gz_df)}, predictions={len(pred_df)}"
            )
        for i in range(min(len(gz_df), len(pred_df))):
            pred_records.append({
                "pair": tuple(sorted([gz_df.iloc[i]["record_id_1"], gz_df.iloc[i]["record_id_2"]])),
                "prediction": int(pred_df.iloc[i]["prediction"]),
                "confidence": pred_df.iloc[i].get("confidence", None),
            })

    # Choose scoring method: LR model or logit-space fallback
    llm_pairs = set()
    llm_confidences = {}
    accepted = 0
    rejected = 0

    if scorer_model_path is not None:
        # --- LR model scoring ---
        model_data = joblib.load(scorer_model_path)
        lr_model = model_data["model"]
        lr_scaler = model_data["scaler"]
        lr_features = model_data["features"]
        logger.info(f"Loaded scorer model from {scorer_model_path}")

        from train_scorer import DEMO_FEATURES  # noqa: E402

        for rec in pred_records:
            pair = rec["pair"]
            confidence = rec["confidence"]
            feat_row = feat_lookup.get(pair)
            if feat_row is None:
                rejected += 1
                continue

            # Build LLM logit
            pred = rec["prediction"]
            if confidence is not None and not (isinstance(confidence, float) and math.isnan(confidence)):
                match_prob = confidence if pred == 1 else (1.0 - confidence)
                match_prob = max(1e-4, min(1.0 - 1e-4, match_prob))
                llm_logit = math.log(match_prob / (1.0 - match_prob))
            else:
                llm_logit = 2.0 if pred == 1 else -2.0

            # Build feature vector in model's expected order
            x = []
            for f in lr_features:
                if f == "llm_logit":
                    x.append(llm_logit)
                else:
                    x.append(float(feat_row[f]))

            x_scaled = lr_scaler.transform(np.array([x]))
            prob = lr_model.predict_proba(x_scaled)[0, 1]

            if prob >= 0.5:
                llm_pairs.add(pair)
                llm_confidences[pair] = confidence
                accepted += 1
            else:
                rejected += 1

        logger.info(f"LR scorer: accepted={accepted}, rejected={rejected}")

    else:
        # --- Logit-space fallback ---
        gz_cfg = cfg.get("gray_zone", {})
        w_demo = gz_cfg.get("w_demo", 0.5)
        w_llm = gz_cfg.get("w_llm", 1.0)
        gz_threshold = gz_cfg.get("threshold", 0.0)

        for rec in pred_records:
            pair = rec["pair"]
            prediction = rec["prediction"]
            confidence = rec["confidence"]

            # Get total_score from features
            feat_row = feat_lookup.get(pair)
            total_score = float(feat_row["total_score"]) if feat_row is not None else None

            # Convert LLM output to match probability
            if confidence is not None and not (isinstance(confidence, float) and math.isnan(confidence)):
                match_prob = confidence if prediction == 1 else (1.0 - confidence)
                match_prob = max(1e-4, min(1.0 - 1e-4, match_prob))
                llm_logit = math.log(match_prob / (1.0 - match_prob))
            else:
                llm_logit = 2.0 if prediction == 1 else -2.0

            if total_score is not None:
                demo_logit = total_score - 5.0
                combined_logit = w_demo * demo_logit + w_llm * llm_logit
            else:
                combined_logit = w_llm * llm_logit

            if combined_logit >= gz_threshold:
                llm_pairs.add(pair)
                llm_confidences[pair] = confidence
                accepted += 1
            else:
                rejected += 1

        logger.info(
            f"Logit scoring (w_demo={w_demo}, w_llm={w_llm}, threshold={gz_threshold}): "
            f"accepted={accepted}, rejected={rejected}"
        )

    # --- Step 3: Combine all matches ---
    all_match_pairs = auto_pairs | llm_pairs
    logger.info(f"Total matches: {len(all_match_pairs)}")

    # --- Step 4: Build Boolean Series on MultiIndex ---
    match_index = pd.MultiIndex.from_tuples(
        list(all_match_pairs), names=["record_id_1", "record_id_2"]
    )
    matches = pd.Series(True, index=match_index)

    # --- Step 5: Create golden records ---
    golden_records_df = create_golden_records(matches, patients_df, cfg)
    logger.info(f"Created {len(golden_records_df)} golden records")

    # --- Step 6: Evaluate ---
    eval_metrics = evaluate_matches(matches, ground_truth_df, patients_df)
    golden_metrics = evaluate_golden_records(golden_records_df, ground_truth_df)

    # --- Step 7: Save outputs ---
    golden_records_df.to_csv(out / "golden_records.csv", index=False)

    # Build all_matches.csv with source info
    match_rows = []
    for pair in sorted(all_match_pairs):
        in_auto = pair in auto_pairs
        in_llm = pair in llm_pairs
        if in_auto and in_llm:
            source = "both"
        elif in_auto:
            source = "auto_match"
        else:
            source = "llm"

        match_rows.append({
            "record_id_1": pair[0],
            "record_id_2": pair[1],
            "source": source,
            "total_score": auto_scores.get(pair),
            "llm_confidence": llm_confidences.get(pair),
        })

    all_matches_df = pd.DataFrame(match_rows)
    all_matches_df.to_csv(out / "all_matches.csv", index=False)

    # Combine metrics
    combined_metrics = {**eval_metrics, **golden_metrics}
    combined_metrics["auto_match_count"] = len(auto_pairs)
    combined_metrics["llm_match_count"] = len(llm_pairs)
    combined_metrics["total_match_count"] = len(all_match_pairs)

    with open(out / "evaluation_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2, default=str)

    logger.info("Golden Records stage complete!")
    logger.info(f"  Precision:       {eval_metrics['precision']:.4f}")
    logger.info(f"  Recall:          {eval_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:        {eval_metrics['f1_score']:.4f}")
    logger.info(f"  Golden records:  {len(golden_records_df)}")
    logger.info(f"  True patients:   {golden_metrics.get('num_true_patients', 'N/A')}")


if __name__ == "__main__":
    main()
