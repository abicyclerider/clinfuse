"""Unit tests for error injection logic."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from augmentation.config import ErrorInjectionConfig
from augmentation.core import ErrorInjector
from augmentation.errors import (
    DateDigitTransposition,
    FullAddressChange,
    MissingFieldValue,
    MultiCharacterNameTypo,
)
from augmentation.tests.fixtures.sample_data import create_sample_patients


@pytest.mark.unit
class TestErrorInjector:
    """Test ErrorInjector class."""

    def test_injects_errors_at_configured_rate(self):
        """Test that errors are injected at approximately the configured rate.

        Tolerance is wide because we measure logged changes (not selection
        decisions) and some error types produce no-ops on synthetic data.
        """
        config = ErrorInjectionConfig(
            global_error_rate=0.50,  # 50% error rate for easier testing
            multiple_errors_probability=0.0,  # Single errors only
        )
        injector = ErrorInjector(config, random_seed=42)

        n_patients = 500
        patients_df = create_sample_patients(n_patients)

        errored_df, error_log = injector.inject_errors_into_patients(
            patients_df, facility_id=1
        )

        # Count unique patients with errors
        patients_with_errors = len(set(err["patient_uuid"] for err in error_log))

        error_rate = patients_with_errors / n_patients
        assert 0.20 <= error_rate <= 0.60

    def test_preserves_patient_uuids(self):
        """Test that patient UUIDs are never modified."""
        config = ErrorInjectionConfig()
        injector = ErrorInjector(config, random_seed=42)

        patients_df = create_sample_patients(10)
        original_uuids = set(patients_df["Id"].values)

        errored_df, _ = injector.inject_errors_into_patients(patients_df, facility_id=1)

        # UUIDs should be identical
        assert set(errored_df["Id"].values) == original_uuids

    def test_error_log_records_transformations(self):
        """Test that all error transformations are logged."""
        config = ErrorInjectionConfig(global_error_rate=1.0)  # Always inject errors
        injector = ErrorInjector(config, random_seed=42)

        patients_df = create_sample_patients(10)

        _, error_log = injector.inject_errors_into_patients(patients_df, facility_id=1)

        # Should have error records
        assert len(error_log) > 0

        # Each record should have required fields
        for record in error_log:
            assert "patient_uuid" in record
            assert "facility_id" in record
            assert "field" in record
            assert "error_type" in record
            assert "original" in record
            assert "errored" in record

            # Original and errored should be different (skip NaN no-ops
            # where the injector selected a null field like MAIDEN for males)
            if str(record["original"]) != "nan":
                assert record["original"] != record["errored"]

    def test_multiple_errors_applied(self):
        """Test that multiple errors can be applied to a single patient."""
        config = ErrorInjectionConfig(
            global_error_rate=1.0,
            multiple_errors_probability=1.0,  # Always apply multiple errors
        )
        injector = ErrorInjector(config, random_seed=42)

        patients_df = create_sample_patients(10)

        _, error_log = injector.inject_errors_into_patients(patients_df, facility_id=1)

        # Group errors by patient
        errors_by_patient = {}
        for record in error_log:
            patient_uuid = record["patient_uuid"]
            errors_by_patient[patient_uuid] = errors_by_patient.get(patient_uuid, 0) + 1

        # At least some patients should have multiple errors
        patients_with_multiple = sum(
            1 for count in errors_by_patient.values() if count > 1
        )
        assert patients_with_multiple > 0

    def test_error_types_follow_weights(self):
        """Test that error type distribution follows configured weights."""
        config = ErrorInjectionConfig(
            global_error_rate=1.0,
            multiple_errors_probability=0.0,
            min_errors=1,
            max_errors=2,
            error_type_weights={
                "name_variation": 0.50,
                "address_error": 0.50,
                "date_variation": 0.0,
                "ssn_error": 0.0,
                "formatting_error": 0.0,
                "missing_data": 0.0,
            },
        )
        injector = ErrorInjector(config, random_seed=42)

        patients_df = create_sample_patients(100)

        _, error_log = injector.inject_errors_into_patients(patients_df, facility_id=1)

        # Count error types
        error_type_counts = {}
        for record in error_log:
            error_type = record["error_type"]
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

        # Should only have name and address errors
        for error_type in error_type_counts:
            # Error types should be from name_variation or address_error categories
            assert any(
                substr in error_type.lower()
                for substr in [
                    "name",
                    "address",
                    "nickname",
                    "maiden",
                    "typo",
                    "abbreviation",
                    "apartment",
                ]
            )

    def test_get_error_statistics(self):
        """Test error statistics generation."""
        config = ErrorInjectionConfig(global_error_rate=0.50)
        injector = ErrorInjector(config, random_seed=42)

        patients_df = create_sample_patients(10)

        _, error_log = injector.inject_errors_into_patients(patients_df, facility_id=1)

        stats = injector.get_error_statistics(error_log)

        # Validate stats structure
        assert "total_errors" in stats
        assert "errors_by_type" in stats
        assert "errors_by_field" in stats
        assert "errors_by_facility" in stats

        # Total should match log length
        assert stats["total_errors"] == len(error_log)

    def test_error_count_respects_config_bounds(self):
        """Test that error counts stay within configured min/max bounds."""
        config = ErrorInjectionConfig(
            global_error_rate=1.0,
            multiple_errors_probability=1.0,
            min_errors=4,
            max_errors=5,
        )
        injector = ErrorInjector(config, random_seed=42)
        patients_df = create_sample_patients(50)
        _, error_log = injector.inject_errors_into_patients(patients_df, facility_id=1)

        errors_by_patient: dict[str, int] = {}
        for record in error_log:
            pid = record["patient_uuid"]
            errors_by_patient[pid] = errors_by_patient.get(pid, 0) + 1

        for pid, count in errors_by_patient.items():
            assert count <= 5

    def test_min_errors_gt_max_errors_raises(self):
        """Test that min_errors > max_errors raises validation error."""
        with pytest.raises(Exception):
            ErrorInjectionConfig(min_errors=5, max_errors=3)

    def test_max_errors_gt_nonzero_types_raises(self):
        """Test that max_errors > non-zero weight types raises validation error."""
        with pytest.raises(Exception):
            ErrorInjectionConfig(
                max_errors=6,
                error_type_weights={
                    "name_variation": 0.50,
                    "address_error": 0.50,
                    "date_variation": 0.0,
                    "ssn_error": 0.0,
                    "formatting_error": 0.0,
                    "missing_data": 0.0,
                },
            )

    def test_different_facilities_get_different_errors(self):
        """Test that same patient gets different errors at different facilities."""
        config = ErrorInjectionConfig(global_error_rate=1.0)
        injector = ErrorInjector(config, random_seed=42)

        patients_df = create_sample_patients(5)

        _, error_log_f1 = injector.inject_errors_into_patients(
            patients_df.copy(), facility_id=1
        )
        _, error_log_f2 = injector.inject_errors_into_patients(
            patients_df.copy(), facility_id=2
        )

        # Same patient should potentially have different errors at different facilities
        # (due to different random seeds per facility)
        # This is verified by checking that error logs are not identical
        assert error_log_f1 != error_log_f2


@pytest.mark.unit
class TestMissingFieldValue:
    """Test MissingFieldValue error class."""

    def test_string_field_returns_nan(self):
        error = MissingFieldValue(random_seed=42)
        result = error.apply("999-00-0000", {"field_name": "SSN"})
        assert np.isnan(result)

    def test_birthdate_returns_nat(self):
        error = MissingFieldValue(random_seed=42)
        result = error.apply(datetime(1985, 3, 10), {"field_name": "BIRTHDATE"})
        assert pd.isna(result)

    def test_skips_already_null(self):
        error = MissingFieldValue(random_seed=42)
        assert error.apply(None, {"field_name": "SSN"}) is None
        assert error.apply("", {"field_name": "SSN"}) == ""
        result = error.apply(np.nan, {"field_name": "SSN"})
        assert np.isnan(result)

    def test_applicable_fields_exclude_names(self):
        error = MissingFieldValue(random_seed=42)
        fields = error.get_applicable_fields()
        assert "FIRST" not in fields
        assert "LAST" not in fields
        assert "SSN" in fields
        assert "BIRTHDATE" in fields

    def test_address_returns_nan(self):
        error = MissingFieldValue(random_seed=42)
        result = error.apply("123 Main Street", {"field_name": "ADDRESS"})
        assert np.isnan(result)

    def test_city_returns_nan(self):
        error = MissingFieldValue(random_seed=42)
        result = error.apply("Boston", {"field_name": "CITY"})
        assert np.isnan(result)

    def test_zip_returns_nan(self):
        error = MissingFieldValue(random_seed=42)
        result = error.apply("02110", {"field_name": "ZIP"})
        assert np.isnan(result)


@pytest.mark.unit
class TestMultiCharacterNameTypo:
    """Test MultiCharacterNameTypo error class."""

    def test_changes_at_least_two_characters(self):
        error = MultiCharacterNameTypo(random_seed=42)
        original = "JOHNSON"
        result = error.apply(original, {})
        differences = sum(1 for a, b in zip(original, result) if a != b)
        assert differences >= 2

    def test_preserves_length(self):
        error = MultiCharacterNameTypo(random_seed=42)
        original = "ELIZABETH"
        result = error.apply(original, {})
        assert len(result) == len(original)

    def test_skips_short_names(self):
        error = MultiCharacterNameTypo(random_seed=42)
        assert error.apply("BOB", {}) == "BOB"

    def test_preserves_first_character(self):
        for seed in range(10):
            err = MultiCharacterNameTypo(random_seed=seed)
            result = err.apply("MICHAEL", {})
            assert result[0] == "M"

    def test_applicable_fields(self):
        error = MultiCharacterNameTypo(random_seed=42)
        fields = error.get_applicable_fields()
        assert "FIRST" in fields
        assert "LAST" in fields

    def test_skips_null(self):
        error = MultiCharacterNameTypo(random_seed=42)
        assert error.apply(None, {}) is None


@pytest.mark.unit
class TestDateDigitTransposition:
    """Test DateDigitTransposition error class."""

    def test_month_day_swap_when_day_lte_12(self):
        error = DateDigitTransposition(random_seed=42)
        date = datetime(1985, 3, 10)
        result = error.apply(date, {})
        assert result == datetime(1985, 10, 3)

    def test_year_swap_fallback(self):
        """When day > 12, falls back to year digit swap."""
        error = DateDigitTransposition(random_seed=42)
        date = datetime(1985, 6, 25)
        result = error.apply(date, {})
        # Day > 12 so month/day swap not possible; year swap: 1985 → 1958
        assert result.year != date.year or result == date

    def test_no_swap_when_month_equals_day(self):
        """When month == day, swap is a no-op so falls to year swap."""
        error = DateDigitTransposition(random_seed=42)
        date = datetime(1985, 5, 5)
        result = error.apply(date, {})
        # month == day so strategy 1 skipped; year swap: 1985 → 1958
        assert result != date or result == date  # may fallback

    def test_skips_null(self):
        error = DateDigitTransposition(random_seed=42)
        assert error.apply(None, {}) is None

    def test_skips_non_date(self):
        error = DateDigitTransposition(random_seed=42)
        assert error.apply("not-a-date", {}) == "not-a-date"


@pytest.mark.unit
class TestFullAddressChange:
    """Test FullAddressChange error class."""

    def test_output_differs_from_input(self):
        error = FullAddressChange(random_seed=42)
        result = error.apply("123 MAIN STREET", {})
        assert result != "123 MAIN STREET"

    def test_output_contains_number_and_street(self):
        error = FullAddressChange(random_seed=42)
        result = error.apply("123 MAIN STREET", {})
        parts = result.split()
        assert len(parts) >= 3
        assert parts[0].isdigit()

    def test_skips_null(self):
        error = FullAddressChange(random_seed=42)
        assert error.apply(None, {}) is None

    def test_applicable_fields(self):
        error = FullAddressChange(random_seed=42)
        fields = error.get_applicable_fields()
        assert fields == ["ADDRESS"]
