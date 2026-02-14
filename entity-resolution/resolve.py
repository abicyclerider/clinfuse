#!/usr/bin/env python3
"""
Resolve stage: blocking → comparison → tiered classification → gray zone pair generation.

Produces:
  auto_matches.csv      — high-confidence matches (score >= auto_match_threshold)
  gray_zone_pairs.csv   — pairs for LLM inference (record_id_1, record_id_2, text)
  features.csv          — all candidate pair features
  resolve_metrics.json  — blocking recall, tier counts, stats
"""

import json
import logging
import sys
from pathlib import Path

import click
import pandas as pd
import yaml

# Add paths for imports inside Docker container (/app)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_script_dir))    # entity-resolution/ -> import src.*
sys.path.insert(0, str(_project_root))  # project root -> import shared.*

from src.data_loader import create_record_id  # noqa: E402
from src.blocking import create_candidate_pairs, evaluate_blocking_recall  # noqa: E402
from src.comparison import build_comparison_features, add_composite_features  # noqa: E402
from shared.data_loader import load_facility_patients  # noqa: E402
from shared.ground_truth import load_ground_truth, add_record_ids_to_ground_truth  # noqa: E402
from shared.medical_records import load_medical_records, get_patient_records  # noqa: E402
from shared.summarize import summarize_diff_friendly_from_records, INSTRUCTION  # noqa: E402

logger = logging.getLogger(__name__)


def discover_run_dir(augmented_dir: str) -> Path:
    """Find the single run_* directory inside augmented_dir."""
    candidates = sorted(Path(augmented_dir).glob("run_*"))
    if not candidates:
        raise FileNotFoundError(f"No run_* directory found in {augmented_dir}")
    if len(candidates) > 1:
        logger.warning(f"Multiple run directories found, using latest: {candidates[-1]}")
    return candidates[-1]


def generate_gray_zone_texts(
    gray_zone_features: pd.DataFrame,
    patients_df: pd.DataFrame,
    medical_records: dict,
) -> pd.DataFrame:
    """Build LLM input texts for gray zone pairs.

    For each pair, summarizes both patients' medical records and formats
    them with the INSTRUCTION template expected by the classifier.
    """
    # Build record_id -> (patient_uuid, facility_id) lookup
    record_lookup = (
        patients_df.set_index("record_id")[["id", "facility_id"]].to_dict("index")
    )

    rows = []
    total = len(gray_zone_features)
    for i, (pair_idx, feat_row) in enumerate(gray_zone_features.iterrows(), 1):
        rid1, rid2 = pair_idx

        info1 = record_lookup.get(rid1)
        info2 = record_lookup.get(rid2)
        if info1 is None or info2 is None:
            logger.warning(f"Skipping pair ({rid1}, {rid2}): record not found")
            continue

        recs1 = get_patient_records(info1["id"], info1["facility_id"], medical_records)
        recs2 = get_patient_records(info2["id"], info2["facility_id"], medical_records)

        summary_a = summarize_diff_friendly_from_records(recs1)
        summary_b = summarize_diff_friendly_from_records(recs2)

        text = INSTRUCTION.format(summary_a=summary_a, summary_b=summary_b)
        rows.append({
            "record_id_1": rid1,
            "record_id_2": rid2,
            "total_score": feat_row["total_score"],
            "text": text,
        })

        if i % 100 == 0 or i == total:
            logger.info(f"  Gray zone text generation: {i}/{total}")

    return pd.DataFrame(rows)


@click.command()
@click.option("--augmented-dir", required=True, type=click.Path(exists=True),
              help="Path to augmented data directory")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Output directory for resolved data")
@click.option("--config", required=True, type=click.Path(exists=True),
              help="Path to matching config YAML")
def main(augmented_dir, output_dir, config):
    """Classical entity resolution: blocking, comparison, tiered classification."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(config) as f:
        cfg = yaml.safe_load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Discover run directory ---
    run_dir = discover_run_dir(augmented_dir)
    logger.info(f"Using run directory: {run_dir}")

    # --- Step 2: Load patients ---
    patients_df = load_facility_patients(str(run_dir))
    patients_df = create_record_id(patients_df)
    logger.info(f"Loaded {len(patients_df)} patient records from "
                f"{patients_df['facility_id'].nunique()} facilities")

    # --- Step 3: Load ground truth ---
    ground_truth_df = load_ground_truth(str(run_dir))
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    # --- Step 4: Blocking ---
    blocking_cfg = cfg.get("blocking", {})
    strategy = blocking_cfg.get("strategy", "aggressive_multipass")
    patients_indexed = patients_df.set_index("record_id")

    candidate_pairs = create_candidate_pairs(patients_indexed, strategy=strategy)

    record_id_mapping = patients_df[["record_id", "facility_id", "id"]]
    blocking_metrics = evaluate_blocking_recall(
        candidate_pairs, ground_truth_df, record_id_mapping
    )

    # --- Step 5: Comparison ---
    features = build_comparison_features(candidate_pairs, patients_indexed, cfg)
    features = add_composite_features(features)
    logger.info(f"Computed {len(features)} feature vectors")

    # --- Step 6: Tiered classification ---
    classify_cfg = cfg.get("classification", {})
    auto_reject_thresh = classify_cfg.get("auto_reject_threshold", 4.0)
    auto_match_thresh = classify_cfg.get("auto_match_threshold", 6.0)

    auto_matches_mask = features["total_score"] >= auto_match_thresh
    gray_zone_mask = (
        (features["total_score"] >= auto_reject_thresh)
        & (features["total_score"] < auto_match_thresh)
    )
    auto_reject_count = int((features["total_score"] < auto_reject_thresh).sum())

    auto_matches = features[auto_matches_mask]
    gray_zone = features[gray_zone_mask]

    logger.info(
        f"Tier split: auto_match={len(auto_matches)}, "
        f"gray_zone={len(gray_zone)}, auto_reject={auto_reject_count}"
    )

    # --- Step 7: Save auto_matches.csv ---
    auto_matches_df = pd.DataFrame({
        "record_id_1": [p[0] for p in auto_matches.index],
        "record_id_2": [p[1] for p in auto_matches.index],
        "total_score": auto_matches["total_score"].values,
    })
    auto_matches_df.to_csv(out / "auto_matches.csv", index=False)
    logger.info(f"Saved {len(auto_matches_df)} auto-matches")

    # --- Step 8: Gray zone pair text generation ---
    logger.info("Loading medical records for gray zone text generation...")
    medical_records = load_medical_records(str(run_dir))

    gray_zone_df = generate_gray_zone_texts(gray_zone, patients_df, medical_records)
    gray_zone_df.to_csv(out / "gray_zone_pairs.csv", index=False)
    logger.info(f"Saved {len(gray_zone_df)} gray zone pairs")

    # --- Step 9: Save features.csv ---
    features_out = features.copy()
    features_out.index.names = ["record_id_1", "record_id_2"]
    features_out.to_csv(out / "features.csv")

    # --- Step 10: Metrics ---
    metrics = {
        "num_records": len(patients_df),
        "num_facilities": int(patients_df["facility_id"].nunique()),
        "candidate_pairs": len(candidate_pairs),
        "blocking_recall": blocking_metrics["blocking_recall"],
        "auto_match_count": len(auto_matches),
        "gray_zone_count": len(gray_zone),
        "auto_reject_count": auto_reject_count,
        "auto_match_threshold": auto_match_thresh,
        "auto_reject_threshold": auto_reject_thresh,
    }

    with open(out / "resolve_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Resolve stage complete!")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
