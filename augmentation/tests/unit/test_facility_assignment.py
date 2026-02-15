"""Unit tests for facility assignment logic."""

import pytest

from augmentation.config import FacilityDistributionConfig
from augmentation.core import FacilityAssigner
from augmentation.tests.fixtures.sample_data import (
    create_sample_encounters,
    create_sample_patients,
)


class TestFacilityAssigner:
    """Test FacilityAssigner class."""

    def test_assigns_all_patients(self):
        """Test that all patients are assigned to facilities."""
        config = FacilityDistributionConfig(num_facilities=5)
        assigner = FacilityAssigner(config, random_seed=42)

        patients_df = create_sample_patients(10)
        encounters_df = create_sample_encounters(patients_df, 5)

        patient_facilities, encounter_facilities = (
            assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )

        # All patients should be assigned
        assert len(patient_facilities) == 10

        # Each patient should have at least 1 facility
        for facilities in patient_facilities.values():
            assert len(facilities) >= 1
            assert len(facilities) <= 5

    def test_assigns_all_encounters(self):
        """Test that all encounters are assigned to facilities."""
        config = FacilityDistributionConfig(num_facilities=5)
        assigner = FacilityAssigner(config, random_seed=42)

        patients_df = create_sample_patients(10)
        encounters_df = create_sample_encounters(patients_df, 5)

        patient_facilities, encounter_facilities = (
            assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )

        # All encounters should be assigned
        assert len(encounter_facilities) == len(encounters_df)

        # All facility IDs should be valid
        for facility_id in encounter_facilities.values():
            assert 1 <= facility_id <= 5

    def test_distribution_follows_weights(self):
        """Test that facility count distribution approximately matches configured weights."""
        config = FacilityDistributionConfig(
            num_facilities=10,
            facility_count_weights={
                1: 0.40,
                2: 0.30,
                3: 0.15,
                4: 0.10,
                5: 0.05,
            },
        )
        assigner = FacilityAssigner(config, random_seed=42)

        # Use larger dataset for better statistical validation
        patients_df = create_sample_patients(100)
        encounters_df = create_sample_encounters(patients_df, 10)

        patient_facilities, _ = assigner.assign_patients_to_facilities(
            patients_df, encounters_df
        )

        # Count patients by number of facilities
        facility_counts = {}
        for facilities in patient_facilities.values():
            count = len(facilities)
            facility_counts[count] = facility_counts.get(count, 0) + 1

        # Check distribution is within tolerance (±15% due to randomness)
        tolerance = 0.15
        assert facility_counts.get(1, 0) / 100 == pytest.approx(0.40, abs=tolerance)
        assert facility_counts.get(2, 0) / 100 == pytest.approx(0.30, abs=tolerance)

    def test_primary_facility_receives_more_encounters(self):
        """Test that primary facility receives ~60% of encounters for multi-facility patients."""
        config = FacilityDistributionConfig(
            num_facilities=5,
            facility_count_weights={
                1: 0.0,
                2: 1.0,
            },  # Force all patients to 2 facilities
            primary_facility_weight=0.60,
        )
        assigner = FacilityAssigner(config, random_seed=42)

        patients_df = create_sample_patients(10)
        encounters_df = create_sample_encounters(
            patients_df, 100
        )  # Many encounters per patient

        patient_facilities, encounter_facilities = (
            assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )

        # For each patient, check primary facility has ~60% of encounters
        for patient_uuid, facilities in patient_facilities.items():
            if len(facilities) < 2:
                continue

            primary_facility = facilities[0]
            patient_encounters = encounters_df[encounters_df["PATIENT"] == patient_uuid]

            # Count encounters at primary facility
            primary_count = sum(
                1
                for enc_id in patient_encounters["Id"]
                if encounter_facilities[enc_id] == primary_facility
            )

            primary_ratio = primary_count / len(patient_encounters)

            # Should be approximately 60% (within 10% tolerance due to rounding)
            assert primary_ratio == pytest.approx(0.60, abs=0.10)

    def test_chronological_ordering(self):
        """Test that encounters are distributed chronologically."""
        config = FacilityDistributionConfig(
            num_facilities=3,
            facility_count_weights={1: 0.0, 2: 1.0},  # Force 2 facilities
        )
        assigner = FacilityAssigner(config, random_seed=42)

        patients_df = create_sample_patients(5)
        encounters_df = create_sample_encounters(patients_df, 20)

        patient_facilities, encounter_facilities = (
            assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )

        # For each patient, verify temporal pattern
        for patient_uuid, facilities in patient_facilities.items():
            if len(facilities) < 2:
                continue

            primary_facility = facilities[0]

            patient_encounters = encounters_df[
                encounters_df["PATIENT"] == patient_uuid
            ].sort_values("START")

            # Get timestamps of primary and secondary encounters
            primary_dates = []
            secondary_dates = []

            for _, enc in patient_encounters.iterrows():
                if encounter_facilities[enc["Id"]] == primary_facility:
                    primary_dates.append(enc["START"])
                else:
                    secondary_dates.append(enc["START"])

            # Primary encounters should tend earlier, but may overlap with secondary
            # due to the 60/40 split (first 60% chronologically go to primary)
            if primary_dates and secondary_dates:
                # Convert to timestamps for median calculation
                import statistics

                primary_timestamps = [d.timestamp() for d in primary_dates]
                secondary_timestamps = [d.timestamp() for d in secondary_dates]

                median_primary = statistics.median(primary_timestamps)
                median_secondary = statistics.median(secondary_timestamps)

                # Median primary date should be earlier than median secondary date
                # (allows for some overlap at the boundary)
                assert median_primary <= median_secondary

    def test_get_facility_statistics(self):
        """Test facility statistics generation."""
        config = FacilityDistributionConfig(num_facilities=5)
        assigner = FacilityAssigner(config, random_seed=42)

        patients_df = create_sample_patients(10)
        encounters_df = create_sample_encounters(patients_df, 5)

        patient_facilities, encounter_facilities = (
            assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )

        stats = assigner.get_facility_statistics(
            patient_facilities, encounter_facilities
        )

        # Validate stats structure
        assert stats["total_patients"] == 10
        assert stats["total_encounters"] == 50
        assert "facility_count_distribution" in stats
        assert "encounter_count_per_facility" in stats
        assert "patient_count_per_facility" in stats
