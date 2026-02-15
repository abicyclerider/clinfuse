"""
Data loading and preparation for entity resolution.

Delegates core loading to shared.data_loader.
"""

import logging

import pandas as pd

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
