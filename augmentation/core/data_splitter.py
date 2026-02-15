"""Data splitting logic to partition records by facility."""

from typing import Dict, List, Set

import pandas as pd

from ..utils import DataHandler


class DataSplitter:
    """Splits Synthea data by facility while maintaining referential integrity."""

    def __init__(self):
        """Initialize data splitter."""
        self.data_handler = DataHandler()

    def split_csvs_by_facility(
        self,
        synthea_csvs: Dict[str, pd.DataFrame],
        patient_facilities: Dict[str, List[int]],
        encounter_facilities: Dict[str, int],
    ) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Partition all CSVs by facility.

        Args:
            synthea_csvs: Dictionary of all Synthea CSV DataFrames
            patient_facilities: Dict[patient_uuid -> List[facility_ids]]
            encounter_facilities: Dict[encounter_uuid -> facility_id]

        Returns:
            Dict[facility_id -> Dict[filename -> DataFrame]]
        """
        # Get list of all facilities
        all_facilities = set()
        for facilities in patient_facilities.values():
            all_facilities.update(facilities)

        facility_csvs = {}

        for facility_id in sorted(all_facilities):
            facility_csvs[facility_id] = self._create_facility_csvs(
                facility_id,
                synthea_csvs,
                patient_facilities,
                encounter_facilities,
            )

        return facility_csvs

    def _create_facility_csvs(
        self,
        facility_id: int,
        synthea_csvs: Dict[str, pd.DataFrame],
        patient_facilities: Dict[str, List[int]],
        encounter_facilities: Dict[str, int],
    ) -> Dict[str, pd.DataFrame]:
        """
        Create all CSV files for a specific facility.

        Args:
            facility_id: Facility identifier
            synthea_csvs: All source CSVs
            patient_facilities: Patient to facilities mapping
            encounter_facilities: Encounter to facility mapping

        Returns:
            Dict[filename -> DataFrame] for this facility
        """
        facility_data = {}

        # Step 1: Get encounters for this facility
        facility_encounters = {
            enc_id
            for enc_id, fac_id in encounter_facilities.items()
            if fac_id == facility_id
        }

        # Step 2: Get patients who have encounters at this facility
        facility_patients = {
            patient_uuid
            for patient_uuid, facilities in patient_facilities.items()
            if facility_id in facilities
        }

        # Step 3: Process each CSV type
        for filename, df in synthea_csvs.items():
            if filename in self.data_handler.REFERENCE_TABLES:
                # Reference tables: copy complete table to all facilities
                facility_data[filename] = df.copy()

            elif filename == "patients.csv":
                # Patients: include all who have encounters at this facility
                facility_data[filename] = df[df["Id"].isin(facility_patients)].copy()

            elif filename == "encounters.csv":
                # Encounters: filter by facility assignment
                facility_data[filename] = df[df["Id"].isin(facility_encounters)].copy()

            elif filename in self.data_handler.ENCOUNTER_LINKED_TABLES:
                # Encounter-linked tables: follow encounter assignment
                facility_data[filename] = df[
                    df["ENCOUNTER"].isin(facility_encounters)
                ].copy()

            elif filename == "claims.csv":
                # Claims: linked via APPOINTMENTID (encounter)
                facility_data[filename] = df[
                    df["APPOINTMENTID"].isin(facility_encounters)
                ].copy()

            elif filename == "claims_transactions.csv":
                # Claims transactions: follow parent claims
                if "claims.csv" in facility_data:
                    facility_claim_ids = set(facility_data["claims.csv"]["Id"].values)
                    facility_data[filename] = df[
                        df["CLAIMID"].isin(facility_claim_ids)
                    ].copy()
                else:
                    facility_data[filename] = pd.DataFrame(columns=df.columns)

            elif filename == "payer_transitions.csv":
                # Payer transitions: temporal split based on encounter date range
                facility_data[filename] = self._split_payer_transitions_temporally(
                    df, facility_patients, facility_data["encounters.csv"]
                )

            else:
                # Unknown table type - skip with warning
                print(
                    f"Warning: Unknown table type {filename}, skipping for facility {facility_id}"
                )
                facility_data[filename] = pd.DataFrame(columns=df.columns)

        return facility_data

    def _split_payer_transitions_temporally(
        self,
        payer_transitions_df: pd.DataFrame,
        facility_patients: Set[str],
        facility_encounters_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Split payer_transitions temporally based on facility's encounter date range.

        Each facility only gets insurance transitions during the time period
        they treated the patient.

        Args:
            payer_transitions_df: Full payer transitions DataFrame
            facility_patients: Set of patient UUIDs at this facility
            facility_encounters_df: Encounters DataFrame for this facility

        Returns:
            Filtered payer transitions DataFrame
        """
        # Filter to relevant patients
        patient_transitions = payer_transitions_df[
            payer_transitions_df["PATIENT"].isin(facility_patients)
        ].copy()

        if len(patient_transitions) == 0 or len(facility_encounters_df) == 0:
            return pd.DataFrame(columns=payer_transitions_df.columns)

        # Calculate encounter date range for each patient at this facility
        patient_date_ranges = (
            facility_encounters_df.groupby("PATIENT")["START"]
            .agg(min_date="min", max_date="max")
            .reset_index()
        )

        # Merge with transitions to get date ranges
        merged = patient_transitions.merge(
            patient_date_ranges, left_on="PATIENT", right_on="PATIENT", how="inner"
        )

        # Filter: Include transitions that overlap with facility's encounter period
        # START_DATE <= max_encounter_date AND (END_DATE >= min_encounter_date OR END_DATE is null)
        mask = (merged["START_DATE"] <= merged["max_date"]) & (
            (merged["END_DATE"] >= merged["min_date"]) | (merged["END_DATE"].isna())
        )

        filtered_transitions = merged[mask].copy()

        # Drop the temporary columns
        filtered_transitions = filtered_transitions.drop(
            columns=["min_date", "max_date"]
        )

        return filtered_transitions
