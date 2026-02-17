"""Data integrity validation utilities."""

from typing import Dict, List, Tuple

import pandas as pd

from .data_handler import DataHandler


class DataValidator:
    """Validates referential integrity and data consistency."""

    def __init__(self):
        """Initialize validator."""
        self.data_handler = DataHandler()

    def validate_facility_tables(
        self, facility_tables: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a facility's tables for referential integrity.

        Args:
            facility_tables: Dictionary of DataFrames for a facility

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Validate patient references
        errors.extend(self._validate_patient_references(facility_tables))

        # Validate encounter references
        errors.extend(self._validate_encounter_references(facility_tables))

        # Validate claims references
        errors.extend(self._validate_claims_references(facility_tables))

        # Validate non-empty critical tables
        errors.extend(self._validate_non_empty_tables(facility_tables))

        return len(errors) == 0, errors

    def _validate_patient_references(
        self, facility_tables: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Validate all PATIENT foreign keys resolve."""
        errors = []

        if "patients" not in facility_tables:
            return ["Missing patients table"]

        patient_ids = set(facility_tables["patients"]["Id"].values)

        # Check encounters
        if (
            "encounters" in facility_tables
            and len(facility_tables["encounters"]) > 0
        ):
            encounter_patients = set(facility_tables["encounters"]["PATIENT"].values)
            invalid = encounter_patients - patient_ids
            if invalid:
                errors.append(
                    f"encounters has {len(invalid)} invalid PATIENT references"
                )

        # Check payer_transitions
        if (
            "payer_transitions" in facility_tables
            and len(facility_tables["payer_transitions"]) > 0
        ):
            transition_patients = set(
                facility_tables["payer_transitions"]["PATIENT"].values
            )
            invalid = transition_patients - patient_ids
            if invalid:
                errors.append(
                    f"payer_transitions has {len(invalid)} invalid PATIENT references"
                )

        return errors

    def _validate_encounter_references(
        self, facility_tables: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Validate all ENCOUNTER foreign keys resolve."""
        errors = []

        if "encounters" not in facility_tables:
            return ["Missing encounters table"]

        encounter_ids = set(facility_tables["encounters"]["Id"].values)

        # Check encounter-linked tables
        for table_name in self.data_handler.ENCOUNTER_LINKED_TABLES:
            if table_name in facility_tables and len(facility_tables[table_name]) > 0:
                df = facility_tables[table_name]
                if "ENCOUNTER" in df.columns:
                    table_encounters = set(df["ENCOUNTER"].values)
                    invalid = table_encounters - encounter_ids
                    if invalid:
                        errors.append(
                            f"{table_name} has {len(invalid)} invalid ENCOUNTER references"
                        )

        # Check claims (uses APPOINTMENTID)
        if "claims" in facility_tables and len(facility_tables["claims"]) > 0:
            claims_encounters = set(facility_tables["claims"]["APPOINTMENTID"].values)
            invalid = claims_encounters - encounter_ids
            if invalid:
                errors.append(
                    f"claims has {len(invalid)} invalid APPOINTMENTID references"
                )

        return errors

    def _validate_claims_references(
        self, facility_tables: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Validate claims_transactions references to claims."""
        errors = []

        if "claims_transactions" not in facility_tables:
            return []

        if len(facility_tables["claims_transactions"]) == 0:
            return []

        if "claims" not in facility_tables:
            errors.append("claims_transactions present but claims missing")
            return errors

        claim_ids = set(facility_tables["claims"]["Id"].values)
        transaction_claims = set(
            facility_tables["claims_transactions"]["CLAIMID"].values
        )

        invalid = transaction_claims - claim_ids
        if invalid:
            errors.append(
                f"claims_transactions has {len(invalid)} invalid CLAIMID references"
            )

        return errors

    def _validate_non_empty_tables(
        self, facility_tables: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Validate that critical tables are not empty."""
        errors = []

        critical_tables = ["patients", "encounters"]

        for table_name in critical_tables:
            if table_name not in facility_tables:
                errors.append(f"Missing critical table: {table_name}")
            elif len(facility_tables[table_name]) == 0:
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
