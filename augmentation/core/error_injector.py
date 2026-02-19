"""Error injection orchestrator."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import ErrorInjectionConfig
from ..errors import (
    AddressAbbreviation,
    ApartmentFormatVariation,
    CapitalizationError,
    DateDigitTransposition,
    DateOffByOne,
    ExtraWhitespace,
    FullAddressChange,
    LeadingTrailingWhitespace,
    MaidenNameUsage,
    MissingFieldValue,
    MultiCharacterNameTypo,
    NameTypo,
    NicknameSubstitution,
    SpecialCharacterVariation,
    SSNDigitError,
    SSNFormatVariation,
    SSNTransposition,
)


class ErrorInjector:
    """Orchestrates error injection into demographic data.

    Note: Current error types represent baseline common data quality issues.
    These can be refined/expanded based on research findings in:
    augmentation/config/research/error_patterns.md

    To add new error types: implement BaseError subclass and register here.
    """

    # Map config error type names to error classes
    ERROR_TYPE_REGISTRY = {
        "name_variation": [
            NicknameSubstitution,
            NameTypo,
            MaidenNameUsage,
            MultiCharacterNameTypo,
        ],
        "address_error": [
            AddressAbbreviation,
            ApartmentFormatVariation,
            FullAddressChange,
        ],
        "date_variation": [
            DateOffByOne,
            DateDigitTransposition,
        ],
        "ssn_error": [
            SSNTransposition,
            SSNDigitError,
            SSNFormatVariation,
        ],
        "formatting_error": [
            CapitalizationError,
            ExtraWhitespace,
            LeadingTrailingWhitespace,
            SpecialCharacterVariation,
        ],
        "missing_data": [
            MissingFieldValue,
        ],
    }

    def __init__(self, config: ErrorInjectionConfig, random_seed: int = 42):
        """
        Initialize error injector.

        Args:
            config: Error injection configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed

        # Prepare error type distribution
        self._prepare_error_type_distribution()

    def _prepare_error_type_distribution(self) -> None:
        """Prepare arrays for weighted random sampling of error types."""
        weights = self.config.error_type_weights
        self.error_types = [k for k, v in weights.items() if v > 0]
        probs = [weights[k] for k in self.error_types]
        total = sum(probs)
        self.error_type_probabilities = [p / total for p in probs]

    def inject_errors_into_patients(
        self,
        patients_df: pd.DataFrame,
        facility_id: int,
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Inject errors into patient demographic data.

        Args:
            patients_df: Patient DataFrame
            facility_id: Facility identifier (used for random seed variation)

        Returns:
            Tuple of (errored DataFrame, list of error records for logging)
        """
        errored_df = patients_df.copy()
        error_log = []

        n = len(errored_df)
        if n == 0:
            return errored_df, error_log

        # Vectorized: decide which patients get errors
        error_rolls = self.rng.random(n)
        gets_error = error_rolls < self.config.global_error_rate

        # Vectorized: decide number of errors for patients that get them
        multi_rolls = self.rng.random(n)
        # Pre-generate integers for multi-error patients
        multi_counts = self.rng.integers(
            self.config.min_errors, self.config.max_errors + 1, size=n
        )

        indices = errored_df.index
        id_values = errored_df["Id"].values

        for pos in np.where(gets_error)[0]:
            idx = indices[pos]
            patient_uuid = id_values[pos]

            # Decide number of errors (using pre-rolled values)
            num_errors = 1
            if multi_rolls[pos] < self.config.multiple_errors_probability:
                num_errors = int(multi_counts[pos])

            # Select error types
            selected_error_types = self.rng.choice(
                self.error_types,
                size=min(num_errors, len(self.error_types)),
                replace=False,
                p=self.error_type_probabilities,
            )

            # Apply errors
            for error_type in selected_error_types:
                error_records = self._apply_error_type(
                    error_type,
                    errored_df,
                    idx,
                    patient_uuid,
                    facility_id,
                )
                error_log.extend(error_records)

        return errored_df, error_log

    def _apply_error_type(
        self,
        error_type: str,
        df: pd.DataFrame,
        row_idx: int,
        patient_uuid: str,
        facility_id: int,
    ) -> List[Dict]:
        """
        Apply a specific error type to a patient record.

        Args:
            error_type: Error type name (from config)
            df: DataFrame to modify in-place
            row_idx: Row index to modify
            patient_uuid: Patient UUID for logging
            facility_id: Facility ID for logging

        Returns:
            List of error log records
        """
        error_log = []

        # Get error classes for this type
        error_classes = self.ERROR_TYPE_REGISTRY.get(error_type, [])
        if not error_classes:
            return error_log

        # Select a random error class from this type
        error_class = self.rng.choice(error_classes)

        # Instantiate error with unique seed
        error_seed = self.random_seed + facility_id * 10000 + row_idx
        error_instance = error_class(random_seed=error_seed)

        # Get applicable fields
        applicable_fields = error_instance.get_applicable_fields()

        # Filter to fields that exist in DataFrame
        existing_fields = [f for f in applicable_fields if f in df.columns]

        if not existing_fields:
            return error_log

        # Choose a random field to modify
        field = self.rng.choice(existing_fields)

        # Get original value
        original_value = df.at[row_idx, field]

        # Apply error — avoid .to_dict() overhead; pass row as Series
        context = {
            "patient_record": df.loc[row_idx],
            "field_name": field,
            "facility_id": facility_id,
        }

        errored_value = error_instance.apply(original_value, context)

        # Only log if value actually changed
        if errored_value != original_value:
            df.at[row_idx, field] = errored_value

            error_log.append(
                {
                    "patient_uuid": patient_uuid,
                    "facility_id": facility_id,
                    "field": field,
                    "error_type": error_instance.get_error_type_name(),
                    "original": str(original_value)
                    if original_value is not None
                    else None,
                    "errored": str(errored_value)
                    if errored_value is not None
                    else None,
                }
            )

        return error_log

    def get_error_statistics(self, error_log: List[Dict]) -> Dict:
        """
        Generate statistics about applied errors.

        Args:
            error_log: List of error records

        Returns:
            Dictionary with error statistics
        """
        if not error_log:
            return {
                "total_errors": 0,
                "errors_by_type": {},
                "errors_by_field": {},
                "errors_by_facility": {},
            }

        # Count by error type
        errors_by_type = {}
        for record in error_log:
            error_type = record["error_type"]
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1

        # Count by field
        errors_by_field = {}
        for record in error_log:
            field = record["field"]
            errors_by_field[field] = errors_by_field.get(field, 0) + 1

        # Count by facility
        errors_by_facility = {}
        for record in error_log:
            facility_id = record["facility_id"]
            errors_by_facility[facility_id] = errors_by_facility.get(facility_id, 0) + 1

        return {
            "total_errors": len(error_log),
            "errors_by_type": errors_by_type,
            "errors_by_field": errors_by_field,
            "errors_by_facility": errors_by_facility,
        }
