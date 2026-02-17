"""Data splitting logic to partition records by facility."""

from typing import Dict, List, Optional, Set

import pandas as pd

from ..utils import DataHandler


class DataSplitter:
    """Splits Synthea data by facility while maintaining referential integrity."""

    def __init__(self):
        """Initialize data splitter."""
        self.data_handler = DataHandler()

    def split_tables_by_facility(
        self,
        synthea_tables: Dict[str, pd.DataFrame],
        patient_facilities: Dict[str, List[int]],
        encounter_facilities: Dict[str, int],
    ) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Partition all tables by facility.

        Args:
            synthea_tables: Dictionary of all Synthea DataFrames
            patient_facilities: Dict[patient_uuid -> List[facility_ids]]
            encounter_facilities: Dict[encounter_uuid -> facility_id]

        Returns:
            Dict[facility_id -> Dict[table_name -> DataFrame]]
        """
        # Get list of all facilities
        all_facilities = set()
        for facilities in patient_facilities.values():
            all_facilities.update(facilities)

        facility_tables = {}

        for facility_id in sorted(all_facilities):
            facility_tables[facility_id] = self.create_facility_tables(
                facility_id,
                synthea_tables,
                patient_facilities,
                encounter_facilities,
            )

        return facility_tables

    def create_facility_tables(
        self,
        facility_id: int,
        synthea_tables: Dict[str, pd.DataFrame],
        patient_facilities: Dict[str, List[int]],
        encounter_facilities: Dict[str, int],
    ) -> Dict[str, pd.DataFrame]:
        """
        Create all tables for a specific facility.

        Args:
            facility_id: Facility identifier
            synthea_tables: All source tables
            patient_facilities: Patient to facilities mapping
            encounter_facilities: Encounter to facility mapping

        Returns:
            Dict[table_name -> DataFrame] for this facility
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

        # Step 3: Process each table type
        for table_name, df in synthea_tables.items():
            if table_name in self.data_handler.REFERENCE_TABLES:
                # Reference tables: copy complete table to all facilities
                facility_data[table_name] = df.copy()

            elif table_name == "patients":
                # Patients: include all who have encounters at this facility
                facility_data[table_name] = df[df["Id"].isin(facility_patients)].copy()

            elif table_name == "encounters":
                # Encounters: filter by facility assignment
                facility_data[table_name] = df[df["Id"].isin(facility_encounters)].copy()

            elif table_name in self.data_handler.ENCOUNTER_LINKED_TABLES:
                # Encounter-linked tables: follow encounter assignment
                facility_data[table_name] = df[
                    df["ENCOUNTER"].isin(facility_encounters)
                ].copy()

            elif table_name == "claims":
                # Claims: linked via APPOINTMENTID (encounter)
                facility_data[table_name] = df[
                    df["APPOINTMENTID"].isin(facility_encounters)
                ].copy()

            elif table_name == "claims_transactions":
                # Claims transactions: follow parent claims
                if "claims" in facility_data:
                    facility_claim_ids = set(facility_data["claims"]["Id"].values)
                    facility_data[table_name] = df[
                        df["CLAIMID"].isin(facility_claim_ids)
                    ].copy()
                else:
                    facility_data[table_name] = pd.DataFrame(columns=df.columns)

            elif table_name == "payer_transitions":
                # Payer transitions: temporal split based on encounter date range
                facility_data[table_name] = self._split_payer_transitions_temporally(
                    df, facility_patients, facility_data["encounters"]
                )

            else:
                # Unknown table type - skip with warning
                print(
                    f"Warning: Unknown table type {table_name}, skipping for facility {facility_id}"
                )
                facility_data[table_name] = pd.DataFrame(columns=df.columns)

        return facility_data

    def filter_table_for_facility(
        self,
        table_name: str,
        df: pd.DataFrame,
        facility_id: int,
        facility_patients: Set[str],
        facility_encounters: Set[str],
        facility_encounters_df: Optional[pd.DataFrame] = None,
        facility_claim_ids: Optional[Set[str]] = None,
        copy: bool = True,
        patient_date_ranges: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Filter a single source table for one facility.

        Same logic as the inner loop of create_facility_tables, but operates on
        one table at a time for streaming use.

        Args:
            table_name: Table name (e.g. 'conditions')
            df: Full source DataFrame for this table
            facility_id: Facility identifier
            facility_patients: Set of patient UUIDs at this facility
            facility_encounters: Set of encounter UUIDs at this facility
            facility_encounters_df: Encounters DataFrame for this facility
                (required for payer_transitions temporal split)
            facility_claim_ids: Set of claim IDs for this facility
                (required for claims_transactions)
            copy: Whether to copy filtered results (default True). Pass False
                when the caller discards the original (e.g. streaming chunks).
            patient_date_ranges: Optional pre-computed date ranges (PATIENT,
                min_date, max_date).

        Returns:
            Filtered DataFrame for this facility
        """

        def _maybe_copy(result: pd.DataFrame) -> pd.DataFrame:
            return result.copy() if copy else result

        if table_name in self.data_handler.REFERENCE_TABLES:
            return _maybe_copy(df)

        elif table_name == "patients":
            return _maybe_copy(df[df["Id"].isin(facility_patients)])

        elif table_name == "encounters":
            return _maybe_copy(df[df["Id"].isin(facility_encounters)])

        elif table_name in self.data_handler.ENCOUNTER_LINKED_TABLES:
            return _maybe_copy(df[df["ENCOUNTER"].isin(facility_encounters)])

        elif table_name == "claims":
            return _maybe_copy(df[df["APPOINTMENTID"].isin(facility_encounters)])

        elif table_name == "claims_transactions":
            if facility_claim_ids:
                return _maybe_copy(df[df["CLAIMID"].isin(facility_claim_ids)])
            else:
                return pd.DataFrame(columns=df.columns)

        elif table_name == "payer_transitions":
            if facility_encounters_df is not None or patient_date_ranges is not None:
                return self._split_payer_transitions_temporally(
                    df,
                    facility_patients,
                    facility_encounters_df,
                    patient_date_ranges=patient_date_ranges,
                )
            else:
                return pd.DataFrame(columns=df.columns)

        else:
            print(
                f"Warning: Unknown table type {table_name}, "
                f"skipping for facility {facility_id}"
            )
            return pd.DataFrame(columns=df.columns)

    def _split_payer_transitions_temporally(
        self,
        payer_transitions_df: pd.DataFrame,
        facility_patients: Set[str],
        facility_encounters_df: pd.DataFrame,
        patient_date_ranges: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Split payer_transitions temporally based on facility's encounter date range.

        Each facility only gets insurance transitions during the time period
        they treated the patient.

        Args:
            payer_transitions_df: Full payer transitions DataFrame
            facility_patients: Set of patient UUIDs at this facility
            facility_encounters_df: Encounters DataFrame for this facility
            patient_date_ranges: Optional pre-computed date ranges (PATIENT,
                min_date, max_date).  When provided, facility_encounters_df
                may be empty/None.

        Returns:
            Filtered payer transitions DataFrame
        """
        # Filter to relevant patients
        patient_transitions = payer_transitions_df[
            payer_transitions_df["PATIENT"].isin(facility_patients)
        ].copy()

        if len(patient_transitions) == 0:
            return pd.DataFrame(columns=payer_transitions_df.columns)

        # Use pre-computed date ranges if available, otherwise compute
        if patient_date_ranges is None:
            if facility_encounters_df is None or len(facility_encounters_df) == 0:
                return pd.DataFrame(columns=payer_transitions_df.columns)
            patient_date_ranges = (
                facility_encounters_df.groupby("PATIENT")["START"]
                .agg(min_date="min", max_date="max")
                .reset_index()
            )

        if len(patient_date_ranges) == 0:
            return pd.DataFrame(columns=payer_transitions_df.columns)

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
