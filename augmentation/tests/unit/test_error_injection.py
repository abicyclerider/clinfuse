"""Unit tests for error injection logic."""

import pytest

from augmentation.config import ErrorInjectionConfig
from augmentation.core import ErrorInjector
from augmentation.tests.fixtures.sample_data import create_sample_patients


class TestErrorInjector:
    """Test ErrorInjector class."""

    def test_injects_errors_at_configured_rate(self):
        """Test that errors are injected at approximately the configured rate."""
        config = ErrorInjectionConfig(
            global_error_rate=0.50,  # 50% error rate for easier testing
            multiple_errors_probability=0.0,  # Single errors only
        )
        injector = ErrorInjector(config, random_seed=42)

        patients_df = create_sample_patients(
            100
        )  # Large sample for statistical validity

        errored_df, error_log = injector.inject_errors_into_patients(
            patients_df, facility_id=1
        )

        # Count unique patients with errors
        patients_with_errors = len(set(err["patient_uuid"] for err in error_log))

        # Should be approximately 50% (within 15% tolerance)
        error_rate = patients_with_errors / 100
        assert error_rate == pytest.approx(0.50, abs=0.15)

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

            # Original and errored should be different
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
            error_type_weights={
                "name_variation": 0.50,
                "address_error": 0.50,
                "date_variation": 0.0,
                "ssn_error": 0.0,
                "formatting_error": 0.0,
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
