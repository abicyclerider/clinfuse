#!/usr/bin/env python3
"""
Resolve stage: Splink probabilistic linkage → tiered classification → gray zone pair generation.

Produces:
  auto_matches.parquet      — high-confidence matches (match_probability >= auto_match_probability)
  gray_zone_pairs.parquet   — pairs for LLM inference (record_id_1, record_id_2, text)
  features.parquet          — all candidate pair features (Splink output)
  resolve_metrics.json      — blocking recall, tier counts, Splink-only benchmarks
"""

import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml

from entity_resolution.core.splink_linker import (
    classify_predictions,
    create_linker,
    evaluate_splink_only,
    predict_matches,
    train_model,
)
from shared.data_loader import create_record_id, load_facility_patients
from shared.ground_truth import (
    add_record_ids_to_ground_truth,
    generate_true_pairs_from_ground_truth,
    load_ground_truth,
)
from shared.medical_records import load_medical_records
from shared.summarize import (
    INSTRUCTION,
    summarize_diff_friendly_from_records,
)

# Only load the record types and columns the summarizer actually uses
SUMMARIZER_RECORD_TYPES = [
    "conditions",
    "medications",
    "allergies",
    "observations",
    "procedures",
]
SUMMARIZER_COLUMNS = {
    "conditions": ["PATIENT", "START", "STOP", "DESCRIPTION"],
    "medications": ["PATIENT", "DESCRIPTION", "START", "STOP"],
    "allergies": ["PATIENT", "DESCRIPTION"],
    "observations": ["PATIENT", "DATE", "DESCRIPTION", "VALUE", "UNITS"],
    "procedures": ["PATIENT", "START", "DESCRIPTION"],
}

logger = logging.getLogger(__name__)


def discover_run_dir(augmented_dir: str) -> Path:
    """Resolve the augmented data directory containing facilities/."""
    p = Path(augmented_dir)
    if not (p / "facilities").is_dir():
        raise FileNotFoundError(f"No facilities/ directory found in {augmented_dir}")
    return p


def generate_gray_zone_texts(
    gray_zone_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    medical_records: dict,
) -> pd.DataFrame:
    """Build LLM input texts for gray zone pairs.

    Accepts a flat DataFrame with record_id_1, record_id_2, total_score columns
    (Splink output format).
    """
    # Build record_id -> (patient_uuid, facility_id) lookup
    record_lookup = patients_df.set_index("record_id")[["id", "facility_id"]].to_dict(
        "index"
    )

    # Pre-index medical records by (PATIENT, facility_id) for O(1) lookups
    # instead of O(n) mask filtering per call
    indexed_records = {}
    for record_type, df in medical_records.items():
        indexed_records[record_type] = {
            key: group for key, group in df.groupby(["PATIENT", "facility_id"])
        }
    del medical_records
    logger.info("Indexed medical records for fast lookup")

    def get_patient_records_indexed(patient_id, facility_id):
        result = {}
        for record_type, idx in indexed_records.items():
            group = idx.get((patient_id, facility_id))
            if group is not None and not group.empty:
                result[record_type] = group
        return result

    rows = []
    total = len(gray_zone_df)
    for i, (_, feat_row) in enumerate(gray_zone_df.iterrows(), 1):
        rid1 = feat_row["record_id_1"]
        rid2 = feat_row["record_id_2"]

        info1 = record_lookup.get(rid1)
        info2 = record_lookup.get(rid2)
        if info1 is None or info2 is None:
            logger.warning(f"Skipping pair ({rid1}, {rid2}): record not found")
            continue

        recs1 = get_patient_records_indexed(info1["id"], info1["facility_id"])
        recs2 = get_patient_records_indexed(info2["id"], info2["facility_id"])

        summary_a = summarize_diff_friendly_from_records(recs1)
        summary_b = summarize_diff_friendly_from_records(recs2)

        text = INSTRUCTION.format(summary_a=summary_a, summary_b=summary_b)
        rows.append(
            {
                "record_id_1": rid1,
                "record_id_2": rid2,
                "total_score": feat_row["total_score"],
                "text": text,
            }
        )

        if i % 100 == 0 or i == total:
            logger.info(f"  Gray zone text generation: {i}/{total}")

    return pd.DataFrame(rows)


def prepare_for_splink(patients_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare patient DataFrame for Splink.

    Ensures:
      - Only standardized lowercase columns are kept (drop uppercase Synthea
        originals like SSN, STATE, ZIP that collide in case-insensitive DuckDB)
      - record_id is a column (not index)
      - birth_year column exists (for blocking)
      - Missing values are NaN (not empty strings or placeholders)
      - birthdate is string (for DateOfBirthComparison)
    """
    # Columns Splink needs
    keep_cols = [
        "record_id",
        "first_name",
        "last_name",
        "address",
        "city",
        "state",
        "zip",
        "ssn",
        "birthdate",
        "facility_id",
        "id",
        "gender",
        "maiden_name",
        "birth_year",
    ]

    df = patients_df.copy()

    # Ensure record_id is a column
    if df.index.name == "record_id":
        df = df.reset_index()

    # Add birth_year for blocking
    if "birth_year" not in df.columns:
        df["birth_year"] = pd.to_datetime(df["birthdate"]).dt.year

    # Convert birthdate to string for Splink DateOfBirthComparison
    if hasattr(df["birthdate"].dtype, "tz") or pd.api.types.is_datetime64_any_dtype(
        df["birthdate"]
    ):
        df["birthdate"] = df["birthdate"].dt.strftime("%Y-%m-%d")

    # Drop uppercase Synthea columns that collide with standardized ones
    # in DuckDB's case-insensitive namespace
    cols_to_keep = [c for c in keep_cols if c in df.columns]
    df = df[cols_to_keep]

    # Replace empty strings and sentinel values with NaN
    for col in ["ssn", "first_name", "last_name", "address", "city", "state", "zip"]:
        if col in df.columns:
            df[col] = df[col].replace({"": np.nan, "000-00-0000": np.nan})

    return df


@click.command()
@click.option(
    "--augmented-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to augmented data directory",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for resolved data",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to matching config YAML",
)
def main(augmented_dir, output_dir, config):
    """Probabilistic entity resolution with Splink v4."""
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
    logger.info(
        f"Loaded {len(patients_df)} patient records from "
        f"{patients_df['facility_id'].nunique()} facilities"
    )

    # --- Step 3: Load ground truth ---
    ground_truth_df = load_ground_truth(str(run_dir))
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    # --- Step 4: Splink — create linker and train model ---
    splink_df = prepare_for_splink(patients_df)
    linker, predict_threshold = create_linker(splink_df, cfg)
    train_model(linker)

    # --- Step 5: Predict and classify ---
    all_predictions = predict_matches(linker, cfg)
    auto_matches, gray_zone, all_predictions = classify_predictions(
        all_predictions, cfg
    )

    logger.info(
        f"Tier split: auto_match={len(auto_matches)}, "
        f"gray_zone={len(gray_zone)}, "
        f"total_predictions={len(all_predictions)}"
    )

    # --- Step 6: Evaluate blocking recall ---
    true_pairs = generate_true_pairs_from_ground_truth(ground_truth_df)
    # Check how many true pairs appear in Splink's predictions
    predicted_pairs_set = set()
    for _, row in all_predictions.iterrows():
        pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
        predicted_pairs_set.add(pair)

    true_pairs_found = sum(1 for p in true_pairs if p in predicted_pairs_set)
    blocking_recall = true_pairs_found / len(true_pairs) if true_pairs else 0
    logger.info(
        f"Blocking recall: {blocking_recall:.2%} "
        f"({true_pairs_found}/{len(true_pairs)} true pairs in predictions)"
    )

    # --- Step 7: Save auto-matches ---
    auto_matches_out = auto_matches[
        ["record_id_1", "record_id_2", "total_score"]
    ].copy()
    auto_matches_out.to_parquet(out / "auto_matches.parquet", index=False)
    logger.info(f"Saved {len(auto_matches_out)} auto-matches")

    # --- Step 8: Gray zone pair text generation ---
    logger.info("Loading medical records for gray zone text generation...")
    medical_records = load_medical_records(
        str(run_dir), record_types=SUMMARIZER_RECORD_TYPES, columns=SUMMARIZER_COLUMNS
    )

    gray_zone_df = generate_gray_zone_texts(gray_zone, patients_df, medical_records)
    gray_zone_df.to_parquet(out / "gray_zone_pairs.parquet", index=False)
    logger.info(f"Saved {len(gray_zone_df)} gray zone pairs")

    # --- Step 9: Save features ---
    all_predictions.to_parquet(out / "features.parquet", index=False)

    # --- Step 10: Splink-only evaluation ---
    splink_metrics = evaluate_splink_only(all_predictions, true_pairs, cfg)

    # --- Step 11: Metrics ---
    splink_cfg = cfg.get("splink", {})
    auto_reject_count = int(
        (
            all_predictions["match_probability"]
            < splink_cfg.get("auto_reject_probability", 0.05)
        ).sum()
    )

    metrics = {
        "num_records": len(patients_df),
        "num_facilities": int(patients_df["facility_id"].nunique()),
        "candidate_pairs": len(all_predictions),
        "blocking_recall": blocking_recall,
        "auto_match_count": len(auto_matches),
        "gray_zone_count": len(gray_zone),
        "auto_reject_count": auto_reject_count,
        "auto_match_probability": splink_cfg.get("auto_match_probability", 0.95),
        "auto_reject_probability": splink_cfg.get("auto_reject_probability", 0.05),
        **splink_metrics,
    }

    with open(out / "resolve_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Resolve stage complete!")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
