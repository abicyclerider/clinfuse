"""
Medical records loader for clinical data from Synthea facilities.

Loads all clinical CSV types (encounters, conditions, medications, etc.)
with facility_id tagging for cross-facility patient matching.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Clinical record types to load (excludes financial: claims, claims_transactions)
CLINICAL_RECORD_TYPES = [
    "encounters",
    "conditions",
    "medications",
    "observations",
    "procedures",
    "immunizations",
    "allergies",
    "careplans",
    "imaging_studies",
    "devices",
    "supplies",
]


def load_medical_records(
    run_dir: str, record_types: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """
    Load clinical record types from all facilities.

    Returns dict keyed by record type (e.g., 'conditions', 'medications').
    Each DataFrame has facility_id + PATIENT columns for joining with patient data.

    Args:
        run_dir: Path to augmentation run directory
        record_types: Record types to load (default: all CLINICAL_RECORD_TYPES)

    Returns:
        Dictionary mapping record type name to DataFrame with all facilities combined
    """
    types_to_load = record_types or CLINICAL_RECORD_TYPES

    run_path = Path(run_dir)
    facilities_dir = run_path / "facilities"

    if not facilities_dir.exists():
        raise FileNotFoundError(f"Facilities directory not found: {facilities_dir}")

    facility_dirs = sorted([d for d in facilities_dir.iterdir() if d.is_dir()])
    logger.info(f"Loading medical records from {len(facility_dirs)} facilities...")

    records = {}

    for record_type in types_to_load:
        frames = []

        for facility_dir in facility_dirs:
            parquet_path = facility_dir / f"{record_type}.parquet"
            csv_path = facility_dir / f"{record_type}.csv"

            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
            elif csv_path.exists():
                df = pd.read_csv(csv_path, low_memory=False)
            else:
                continue
            df["facility_id"] = facility_dir.name
            frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            records[record_type] = combined
            logger.debug(
                f"  {record_type}: {len(combined):,} rows from {len(frames)} facilities"
            )
        else:
            logger.warning(f"  {record_type}: no data found")

    total_rows = sum(len(df) for df in records.values())
    logger.info(f"Loaded {len(records)} record types, {total_rows:,} total rows")

    return records


def get_patient_records(
    patient_id: str, facility_id: str, medical_records: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """
    Filter medical records to a single patient at a single facility.

    Args:
        patient_id: Patient UUID (the PATIENT column in Synthea CSVs)
        facility_id: Facility identifier (e.g., 'facility_001')
        medical_records: Full medical records dict from load_medical_records()

    Returns:
        Dictionary mapping record type to patient-specific DataFrame
    """
    patient_records = {}

    for record_type, df in medical_records.items():
        mask = (df["PATIENT"] == patient_id) & (df["facility_id"] == facility_id)
        patient_df = df[mask]
        if not patient_df.empty:
            patient_records[record_type] = patient_df

    return patient_records
