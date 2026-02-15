"""Data integrity validation utilities."""

from typing import Dict, List, Tuple

import pandas as pd

from .data_handler import DataHandler


class DataValidator:
    """Validates referential integrity and data consistency."""

    def __init__(self):
        """Initialize validator."""
        self.data_handler = DataHandler()

    def validate_facility_csvs(
        self, facility_csvs: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a facility's CSV files for referential integrity.

        Args:
            facility_csvs: Dictionary of CSV DataFrames for a facility

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Validate patient references
        errors.extend(self._validate_patient_references(facility_csvs))

        # Validate encounter references
        errors.extend(self._validate_encounter_references(facility_csvs))

        # Validate claims references
        errors.extend(self._validate_claims_references(facility_csvs))

        # Validate non-empty critical tables
        errors.extend(self._validate_non_empty_tables(facility_csvs))

        return len(errors) == 0, errors

    def _validate_patient_references(
        self, facility_csvs: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Validate all PATIENT foreign keys resolve."""
        errors = []

        if "patients.csv" not in facility_csvs:
            return ["Missing patients.csv"]

        patient_ids = set(facility_csvs["patients.csv"]["Id"].values)

        # Check encounters
        if (
            "encounters.csv" in facility_csvs
            and len(facility_csvs["encounters.csv"]) > 0
        ):
            encounter_patients = set(facility_csvs["encounters.csv"]["PATIENT"].values)
            invalid = encounter_patients - patient_ids
            if invalid:
                errors.append(
                    f"encounters.csv has {len(invalid)} invalid PATIENT references"
                )

        # Check payer_transitions
        if (
            "payer_transitions.csv" in facility_csvs
            and len(facility_csvs["payer_transitions.csv"]) > 0
        ):
            transition_patients = set(
                facility_csvs["payer_transitions.csv"]["PATIENT"].values
            )
            invalid = transition_patients - patient_ids
            if invalid:
                errors.append(
                    f"payer_transitions.csv has {len(invalid)} invalid PATIENT references"
                )

        return errors

    def _validate_encounter_references(
        self, facility_csvs: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Validate all ENCOUNTER foreign keys resolve."""
        errors = []

        if "encounters.csv" not in facility_csvs:
            return ["Missing encounters.csv"]

        encounter_ids = set(facility_csvs["encounters.csv"]["Id"].values)

        # Check encounter-linked tables
        for table_name in self.data_handler.ENCOUNTER_LINKED_TABLES:
            if table_name in facility_csvs and len(facility_csvs[table_name]) > 0:
                df = facility_csvs[table_name]
                if "ENCOUNTER" in df.columns:
                    table_encounters = set(df["ENCOUNTER"].values)
                    invalid = table_encounters - encounter_ids
                    if invalid:
                        errors.append(
                            f"{table_name} has {len(invalid)} invalid ENCOUNTER references"
                        )

        # Check claims (uses APPOINTMENTID)
        if "claims.csv" in facility_csvs and len(facility_csvs["claims.csv"]) > 0:
            claims_encounters = set(facility_csvs["claims.csv"]["APPOINTMENTID"].values)
            invalid = claims_encounters - encounter_ids
            if invalid:
                errors.append(
                    f"claims.csv has {len(invalid)} invalid APPOINTMENTID references"
                )

        return errors

    def _validate_claims_references(
        self, facility_csvs: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Validate claims_transactions references to claims."""
        errors = []

        if "claims_transactions.csv" not in facility_csvs:
            return []

        if len(facility_csvs["claims_transactions.csv"]) == 0:
            return []

        if "claims.csv" not in facility_csvs:
            errors.append("claims_transactions.csv present but claims.csv missing")
            return errors

        claim_ids = set(facility_csvs["claims.csv"]["Id"].values)
        transaction_claims = set(
            facility_csvs["claims_transactions.csv"]["CLAIMID"].values
        )

        invalid = transaction_claims - claim_ids
        if invalid:
            errors.append(
                f"claims_transactions.csv has {len(invalid)} invalid CLAIMID references"
            )

        return errors

    def _validate_non_empty_tables(
        self, facility_csvs: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Validate that critical tables are not empty."""
        errors = []

        critical_tables = ["patients.csv", "encounters.csv"]

        for table_name in critical_tables:
            if table_name not in facility_csvs:
                errors.append(f"Missing critical table: {table_name}")
            elif len(facility_csvs[table_name]) == 0:
                errors.append(f"Critical table is empty: {table_name}")

        return errors

    def validate_distribution_statistics(
        self,
        stats: Dict,
        expected_total_patients: int,
        expected_total_encounters: int,
        config_weights: Dict[int, float],
        tolerance: float = 0.10,
    ) -> Tuple[bool, List[str]]:
        """
        Validate facility distribution statistics match configuration.

        Args:
            stats: Statistics from FacilityAssigner.get_facility_statistics()
            expected_total_patients: Expected number of patients
            expected_total_encounters: Expected number of encounters
            config_weights: Configured facility count weights
            tolerance: Acceptable deviation from expected distribution (default 10%)

        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []

        # Check total counts
        if stats["total_patients"] != expected_total_patients:
            warnings.append(
                f"Patient count mismatch: expected {expected_total_patients}, "
                f"got {stats['total_patients']}"
            )

        if stats["total_encounters"] != expected_total_encounters:
            warnings.append(
                f"Encounter count mismatch: expected {expected_total_encounters}, "
                f"got {stats['total_encounters']}"
            )

        # Check facility count distribution
        actual_distribution = stats["facility_count_distribution"]
        total_patients = stats["total_patients"]

        for count, expected_prob in config_weights.items():
            actual_count = actual_distribution.get(count, 0)
            actual_prob = actual_count / total_patients if total_patients > 0 else 0

            lower_bound = expected_prob - tolerance
            upper_bound = expected_prob + tolerance

            if not (lower_bound <= actual_prob <= upper_bound):
                warnings.append(
                    f"Distribution for {count} facilities: expected {expected_prob:.2%}, "
                    f"got {actual_prob:.2%} (outside {tolerance:.0%} tolerance)"
                )

        return len(warnings) == 0, warnings
