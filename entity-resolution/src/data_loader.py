"""
Data loading and preparation for entity resolution.

Loads patient records from multiple facility CSVs and ground truth for validation.
Delegates core loading to shared.data_loader.
"""

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Add project root to path so shared module is importable
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging  # noqa: E402

from shared.data_loader import (  # noqa: E402
    get_run_directory,
    load_facility_patients,
    standardize_columns,
)
from shared.ground_truth import load_ground_truth  # noqa: E402

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
prepare_for_matching = standardize_columns


def create_record_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a unique record identifier for each patient-facility combination.

    Args:
        df: Patient DataFrame with facility_id and id columns

    Returns:
        DataFrame with added record_id column
    """
    df = df.copy()
    df["record_id"] = df["facility_id"] + "_" + df["id"].astype(str)
    return df


def load_data_for_matching(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all necessary data for entity resolution from configuration.

    Args:
        config: Configuration dictionary with base_dir and run_id

    Returns:
        Tuple of (patient_records_df, ground_truth_df)
    """
    run_dir = get_run_directory(config["base_dir"], config["run_id"])

    patients_df = load_facility_patients(str(run_dir))
    patients_df = create_record_id(patients_df)

    ground_truth_df = load_ground_truth(str(run_dir))

    logger.info(
        f"Loaded {len(patients_df)} patient records and {len(ground_truth_df)} ground truth entries"
    )

    return patients_df, ground_truth_df
