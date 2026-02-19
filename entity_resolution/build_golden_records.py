#!/usr/bin/env python3
"""
Golden Records stage: combine auto-matches + LLM predictions -> golden records -> evaluation.

Produces:
  golden_records.parquet     — master patient index with facility provenance
  all_matches.parquet        — all matched pairs with source (auto_match/llm) and scores
  evaluation_metrics.json    — precision, recall, F1
"""

import json
import logging
import math
from pathlib import Path

import click
import pandas as pd
import yaml

from entity_resolution.core.evaluation import evaluate_golden_records, evaluate_matches
from entity_resolution.core.golden_record import create_golden_records
from entity_resolution.core.splink_linker import splink_logit
from shared.data_loader import create_record_id, load_facility_patients
from shared.ground_truth import (
    add_record_ids_to_ground_truth,
    load_ground_truth,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--augmented-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to augmented data directory",
)
@click.option(
    "--auto-matches",
    required=True,
    type=click.Path(exists=True),
    help="Path to auto_matches.csv from resolve stage",
)
@click.option(
    "--predictions",
    required=True,
    type=click.Path(exists=True),
    help="Path to predictions.csv from infer stage",
)
@click.option(
    "--gray-zone-pairs",
    required=False,
    default=None,
    type=click.Path(exists=True),
    help="Path to gray_zone_pairs.csv (only needed if predictions.csv lacks record_id columns)",
)
@click.option(
    "--features",
    "features_path",
    required=False,
    default=None,
    type=click.Path(exists=True),
    help="Path to features.csv from resolve stage (Splink output)",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for golden records",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to matching config YAML",
)
def main(
    augmented_dir,
    auto_matches,
    predictions,
    gray_zone_pairs,
    features_path,
    output_dir,
    config,
):
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

    # Resolve augmented data directory
    run_dir = Path(augmented_dir)
    if not (run_dir / "facilities").is_dir():
        raise FileNotFoundError(f"No facilities/ directory found in {augmented_dir}")
    logger.info(f"Using augmented data: {run_dir}")

    # Load patient data
    patients_df = load_facility_patients(str(run_dir))
    patients_df = create_record_id(patients_df)
    logger.info(f"Loaded {len(patients_df)} patient records")

    # Load ground truth
    ground_truth_df = load_ground_truth(str(run_dir))
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    # --- Step 1: Load auto-matches ---
    auto_df = pd.read_parquet(auto_matches)
    auto_pairs = set()
    auto_scores = {}
    for _, row in auto_df.iterrows():
        pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
        auto_pairs.add(pair)
        auto_scores[pair] = row["total_score"]
    logger.info(f"Auto-matches: {len(auto_pairs)}")

    # --- Step 2: Score gray zone predictions using Splink probability + LLM logit ---
    pred_df = pd.read_parquet(predictions)

    # Load features for Splink match probabilities
    feat_lookup = {}
    if features_path is not None:
        feat_df = pd.read_parquet(features_path)
        for _, row in feat_df.iterrows():
            pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
            feat_lookup[pair] = row
        logger.info(f"Loaded {len(feat_lookup)} feature vectors from {features_path}")

    # Parse predictions into per-pair records
    pred_records = []
    if "record_id_1" in pred_df.columns and "record_id_2" in pred_df.columns:
        for _, row in pred_df.iterrows():
            pred_records.append(
                {
                    "pair": tuple(sorted([row["record_id_1"], row["record_id_2"]])),
                    "prediction": int(row["prediction"]),
                    "confidence": row.get("confidence"),
                }
            )
    else:
        if gray_zone_pairs is None:
            raise click.UsageError(
                "--gray-zone-pairs required when predictions lack record_id columns"
            )
        gz_df = pd.read_parquet(gray_zone_pairs)
        if len(gz_df) != len(pred_df):
            logger.warning(
                f"Row count mismatch: gray_zone_pairs={len(gz_df)}, predictions={len(pred_df)}"
            )
        for i in range(min(len(gz_df), len(pred_df))):
            pred_records.append(
                {
                    "pair": tuple(
                        sorted(
                            [gz_df.iloc[i]["record_id_1"], gz_df.iloc[i]["record_id_2"]]
                        )
                    ),
                    "prediction": int(pred_df.iloc[i]["prediction"]),
                    "confidence": pred_df.iloc[i].get("confidence", None),
                }
            )

    # Splink probability + LLM logit combination
    gz_cfg = cfg.get("gray_zone", {})
    w_splink = gz_cfg.get("w_splink", 1.0)
    w_llm = gz_cfg.get("w_llm", 1.0)
    gz_threshold = gz_cfg.get("threshold", 0.0)
    min_splink_prob = gz_cfg.get("min_splink_probability", 0.0)

    # Bayesian prior correction: LLM was trained on balanced data (50/50) but
    # the gray zone has a much lower true-match rate (~2.5%).  Shift the LLM
    # logit by the difference in log-prior-odds so its output reflects the
    # production base rate.
    llm_train_prior = gz_cfg.get("llm_training_prior", 0.5)
    gz_prior = gz_cfg.get("gray_zone_prior", 0.5)  # 0.5 = no correction
    prior_correction = math.log(gz_prior / (1.0 - gz_prior)) - math.log(
        llm_train_prior / (1.0 - llm_train_prior)
    )
    logger.info(f"LLM prior correction: {prior_correction:.3f} logits "
                f"(train={llm_train_prior}, gz={gz_prior})")

    llm_pairs = set()
    llm_confidences = {}
    accepted = 0
    rejected = 0
    rejected_floor = 0

    for rec in pred_records:
        pair = rec["pair"]
        prediction = rec["prediction"]
        confidence = rec["confidence"]

        # Convert LLM output to match probability -> logit
        if confidence is not None and not (
            isinstance(confidence, float) and math.isnan(confidence)
        ):
            match_prob = confidence if prediction == 1 else (1.0 - confidence)
            match_prob = max(1e-4, min(1.0 - 1e-4, match_prob))
            llm_logit = math.log(match_prob / (1.0 - match_prob))
        else:
            llm_logit = 2.0 if prediction == 1 else -2.0

        # Apply Bayesian prior correction to LLM logit
        llm_logit_corrected = llm_logit + prior_correction

        # Get Splink match probability from features
        feat_row = feat_lookup.get(pair)
        if feat_row is not None and "match_probability" in feat_row.index:
            splink_prob = float(feat_row["match_probability"])
            s_logit = splink_logit(splink_prob)
            combined_logit = w_splink * s_logit + w_llm * llm_logit_corrected
        else:
            splink_prob = None
            combined_logit = w_llm * llm_logit_corrected

        # Reject if Splink probability is below floor (demographics say no match)
        if (
            min_splink_prob > 0
            and splink_prob is not None
            and splink_prob < min_splink_prob
        ):
            rejected += 1
            rejected_floor += 1
            continue

        if combined_logit >= gz_threshold:
            llm_pairs.add(pair)
            llm_confidences[pair] = confidence
            accepted += 1
        else:
            rejected += 1

    logger.info(
        f"Splink+LLM scoring (w_splink={w_splink}, w_llm={w_llm}, "
        f"threshold={gz_threshold}, min_splink_P={min_splink_prob}): "
        f"accepted={accepted}, rejected={rejected} "
        f"({rejected_floor} below Splink floor)"
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
    golden_records_df.to_parquet(out / "golden_records.parquet", index=False)

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

        match_rows.append(
            {
                "record_id_1": pair[0],
                "record_id_2": pair[1],
                "source": source,
                "total_score": auto_scores.get(pair),
                "llm_confidence": llm_confidences.get(pair),
            }
        )

    all_matches_df = pd.DataFrame(match_rows)
    all_matches_df.to_parquet(out / "all_matches.parquet", index=False)

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
