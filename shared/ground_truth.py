"""
Shared ground truth utilities for entity resolution.

Loads ground truth and generates true matching pairs.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_ground_truth(run_dir: str) -> pd.DataFrame:
    """
    Load ground truth mapping of patient records to true patient IDs.

    Args:
        run_dir: Path to augmentation run directory

    Returns:
        DataFrame with columns: facility_id, patient_id, true_patient_id
    """
    run_path = Path(run_dir)
    ground_truth_file = run_path / "metadata" / "ground_truth.csv"

    if not ground_truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

    df = pd.read_csv(ground_truth_file)

    # Normalize facility_id to match patient data format (e.g., 2 → "facility_002")
    if "facility_id" in df.columns:
        df["facility_id"] = df["facility_id"].apply(lambda x: f"facility_{int(x):03d}")

    logger.info(f"Loaded ground truth with {len(df)} records")

    return df


def generate_true_pairs_from_ground_truth(ground_truth: pd.DataFrame) -> set:
    """
    Generate set of true matching pairs from ground truth.

    Args:
        ground_truth: DataFrame with true_patient_id and record_id columns

    Returns:
        Set of tuples (record_id_1, record_id_2) for true matches
    """
    true_id_col = (
        "true_patient_id"
        if "true_patient_id" in ground_truth.columns
        else "original_patient_uuid"
    )

    true_pairs = set()

    for true_id, group in ground_truth.groupby(true_id_col):
        record_ids = (
            group["record_id"].dropna().tolist() if "record_id" in group.columns else []
        )

        for i in range(len(record_ids)):
            for j in range(i + 1, len(record_ids)):
                pair = tuple(sorted([record_ids[i], record_ids[j]]))
                true_pairs.add(pair)

    return true_pairs


def add_record_ids_to_ground_truth(
    ground_truth_df: pd.DataFrame, patients_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add record_id column to ground truth by merging with patient data.

    Args:
        ground_truth_df: Ground truth DataFrame
        patients_df: Patient DataFrame with record_id

    Returns:
        Ground truth DataFrame with record_id column
    """
    mapping = patients_df[["facility_id", "id", "record_id"]].copy()

    gt_with_records = ground_truth_df.merge(
        mapping,
        left_on=["facility_id", "original_patient_uuid"],
        right_on=["facility_id", "id"],
        how="left",
    )

    if "id" in gt_with_records.columns:
        gt_with_records = gt_with_records.drop("id", axis=1)

    gt_with_records = gt_with_records.rename(
        columns={"original_patient_uuid": "true_patient_id"}
    )

    return gt_with_records
