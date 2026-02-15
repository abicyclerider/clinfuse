"""
Shared data loading utilities for entity resolution.

Loads patient records from multiple facility CSVs.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_facility_patients(run_dir: str) -> pd.DataFrame:
    """
    Load patient records from all facility CSVs into a single DataFrame.

    Args:
        run_dir: Path to augmentation run directory (e.g., output/augmented/run_20260202_122731)

    Returns:
        DataFrame with all patient records, including facility_id column
    """
    run_path = Path(run_dir)
    facilities_dir = run_path / "facilities"

    if not facilities_dir.exists():
        raise FileNotFoundError(f"Facilities directory not found: {facilities_dir}")

    all_patients = []
    facility_dirs = sorted([d for d in facilities_dir.iterdir() if d.is_dir()])

    logger.info(f"Loading patients from {len(facility_dirs)} facilities...")

    for facility_dir in facility_dirs:
        facility_id = facility_dir.name
        parquet_file = facility_dir / "patients.parquet"
        csv_file = facility_dir / "patients.csv"

        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
        elif csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            logger.warning(f"No patients file found in {facility_id}")
            continue
        df["facility_id"] = facility_id
        all_patients.append(df)
        logger.debug(f"Loaded {len(df)} patients from {facility_id}")

    if not all_patients:
        raise ValueError("No patient records found")

    combined_df = pd.concat(all_patients, ignore_index=True)
    logger.info(
        f"Loaded {len(combined_df)} total patient records from {len(facility_dirs)} facilities"
    )

    combined_df = standardize_columns(combined_df)

    return combined_df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize patient data columns for matching.

    Renames Synthea columns to lowercase, normalizes names/SSN/dates/ZIP.

    Args:
        df: Raw patient DataFrame

    Returns:
        Cleaned DataFrame ready for entity resolution
    """
    df = df.copy()

    # Strip whitespace from string columns
    string_cols = df.select_dtypes(include=["object"]).columns
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Create standardized field names for matching
    field_mapping = {
        "Id": "id",
        "BIRTHDATE": "birthdate",
        "SSN": "ssn",
        "FIRST": "first_name",
        "LAST": "last_name",
        "MAIDEN": "maiden_name",
        "ADDRESS": "address",
        "CITY": "city",
        "STATE": "state",
        "ZIP": "zip",
        "GENDER": "gender",
    }

    for old_name, new_name in field_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Normalize case for names (title case)
    for col in ["first_name", "last_name", "maiden_name"]:
        if col in df.columns:
            df[col] = df[col].str.title()

    # Normalize SSN format (remove dashes if present)
    if "ssn" in df.columns:
        df["ssn"] = df["ssn"].str.replace("-", "", regex=False)

    # Convert birthdate to datetime
    if "birthdate" in df.columns:
        df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
        df["birth_year"] = df["birthdate"].dt.year

    # Normalize ZIP codes (ensure 5 digits)
    if "zip" in df.columns:
        df["zip"] = df["zip"].astype(str).str.zfill(5)

    # Uppercase state codes
    if "state" in df.columns:
        df["state"] = df["state"].str.upper()

    logger.debug(f"Prepared {len(df)} records for matching")

    return df


def get_run_directory(base_dir: str, run_id: str) -> Path:
    """
    Get the full path to a run directory.

    Args:
        base_dir: Base directory of the project
        run_id: Run identifier (e.g., run_20260202_122731)

    Returns:
        Path to run directory
    """
    run_dir = Path(base_dir) / "output" / "augmented" / run_id

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    return run_dir
