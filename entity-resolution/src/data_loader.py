"""
Data loading and preparation for entity resolution.

Delegates core loading to shared.data_loader.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path so shared module is importable
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging  # noqa: E402

logger = logging.getLogger(__name__)


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
