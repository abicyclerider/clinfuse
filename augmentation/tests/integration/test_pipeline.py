"""Integration tests for the full augmentation pipeline."""

import shutil
import tempfile
from pathlib import Path

import pytest

from augmentation.config import (
    AugmentationConfig,
    ErrorInjectionConfig,
    FacilityDistributionConfig,
    PathConfig,
)
from augmentation.core import (
    CSVSplitter,
    ErrorInjector,
    FacilityAssigner,
    GroundTruthTracker,
)
from augmentation.generators import FacilityGenerator
from augmentation.tests.fixtures.sample_data import create_sample_synthea_csvs
from augmentation.utils import CSVHandler, DataValidator


class TestAugmentationPipeline:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_csvs(self):
        """Create sample Synthea CSVs."""
        return create_sample_synthea_csvs(num_patients=10, encounters_per_patient=10)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        # Create input directory
        input_dir = temp_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        return AugmentationConfig(
            facility_distribution=FacilityDistributionConfig(
                num_facilities=3,
                facility_count_weights={1: 0.5, 2: 0.5},
            ),
            error_injection=ErrorInjectionConfig(
                global_error_rate=0.50,
            ),
            paths=PathConfig(
                input_dir=input_dir,
                output_dir=temp_dir / "output",
            ),
            random_seed=42,
        )

    def test_end_to_end_pipeline(self, sample_csvs, config, temp_dir):
        """Test the complete augmentation pipeline."""
        # Setup
        patients_df = sample_csvs["patients.csv"]
        encounters_df = sample_csvs["encounters.csv"]

        # Step 1: Generate facilities
        facility_generator = FacilityGenerator(random_seed=42)
        facilities_df = facility_generator.generate_facilities(
            config.facility_distribution.num_facilities,
            sample_csvs["organizations.csv"],
        )

        assert len(facilities_df) == 3

        # Step 2: Assign patients to facilities
        facility_assigner = FacilityAssigner(
            config.facility_distribution, random_seed=42
        )
        patient_facilities, encounter_facilities = (
            facility_assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )

        assert len(patient_facilities) == 10
        assert len(encounter_facilities) == 100

        # Step 3: Split CSVs by facility
        csv_splitter = CSVSplitter()
        facility_csvs = csv_splitter.split_csvs_by_facility(
            sample_csvs, patient_facilities, encounter_facilities
        )

        # Should have 3 facilities
        assert len(facility_csvs) == 3

        # Each facility should have all 18 CSV files
        for facility_id, csvs in facility_csvs.items():
            assert "patients.csv" in csvs
            assert "encounters.csv" in csvs
            assert "conditions.csv" in csvs
            assert len(csvs) == 18  # All Synthea CSVs

        # Step 4: Inject errors
        error_injector = ErrorInjector(config.error_injection, random_seed=42)
        ground_truth_tracker = GroundTruthTracker()

        for facility_id, csvs in facility_csvs.items():
            errored_patients_df, error_log = error_injector.inject_errors_into_patients(
                csvs["patients.csv"], facility_id
            )

            facility_csvs[facility_id]["patients.csv"] = errored_patients_df

            # Track ground truth
            for _, patient in errored_patients_df.iterrows():
                patient_uuid = patient["Id"]
                num_encounters = len(
                    csvs["encounters.csv"][
                        csvs["encounters.csv"]["PATIENT"] == patient_uuid
                    ]
                )
                patient_errors = [
                    err["error_type"]
                    for err in error_log
                    if err["patient_uuid"] == patient_uuid
                ]

                ground_truth_tracker.add_patient_facility_mapping(
                    patient_uuid,
                    facility_id,
                    num_encounters,
                    patient.to_dict(),
                    patient_errors,
                )

            ground_truth_tracker.add_error_records(error_log)

        # Step 5: Validate referential integrity
        validator = DataValidator()

        for facility_id, csvs in facility_csvs.items():
            is_valid, errors = validator.validate_facility_csvs(csvs)
            assert is_valid, f"Facility {facility_id} validation failed: {errors}"

        # Step 6: Write outputs
        csv_handler = CSVHandler()
        output_dir = temp_dir / "output"

        for facility_id, csvs in facility_csvs.items():
            csv_handler.write_facility_csvs(
                csvs, output_dir / "facilities", facility_id
            )

        # Verify output files exist
        for facility_id in range(1, 4):
            facility_dir = output_dir / "facilities" / f"facility_{facility_id:03d}"
            assert facility_dir.exists()
            assert (facility_dir / "patients.csv").exists()
            assert (facility_dir / "encounters.csv").exists()

        # Step 7: Export ground truth
        metadata_dir = output_dir / "metadata"
        ground_truth_tracker.export_ground_truth_csv(metadata_dir / "ground_truth.csv")
        ground_truth_tracker.export_error_log_jsonl(metadata_dir / "error_log.jsonl")

        assert (metadata_dir / "ground_truth.csv").exists()
        assert (metadata_dir / "error_log.jsonl").exists()

        # Verify ground truth contains all patients
        ground_truth_stats = ground_truth_tracker.generate_statistics()
        assert ground_truth_stats["unique_patients"] == 10

    def test_referential_integrity_maintained(self, sample_csvs, config):
        """Test that referential integrity is maintained across the pipeline."""
        patients_df = sample_csvs["patients.csv"]
        encounters_df = sample_csvs["encounters.csv"]

        # Run through pipeline
        facility_assigner = FacilityAssigner(
            config.facility_distribution, random_seed=42
        )
        patient_facilities, encounter_facilities = (
            facility_assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )

        csv_splitter = CSVSplitter()
        facility_csvs = csv_splitter.split_csvs_by_facility(
            sample_csvs, patient_facilities, encounter_facilities
        )

        # Validate each facility
        validator = DataValidator()

        for facility_id, csvs in facility_csvs.items():
            is_valid, errors = validator.validate_facility_csvs(csvs)
            assert is_valid, (
                f"Facility {facility_id} has referential integrity issues: {errors}"
            )

            # Manually check key relationships
            patient_ids = set(csvs["patients.csv"]["Id"].values)
            encounter_patients = set(csvs["encounters.csv"]["PATIENT"].values)

            # All encounter patients must be in patients table
            assert encounter_patients.issubset(patient_ids)

            # All conditions must reference valid encounters
            if len(csvs["conditions.csv"]) > 0:
                condition_encounters = set(csvs["conditions.csv"]["ENCOUNTER"].values)
                encounter_ids = set(csvs["encounters.csv"]["Id"].values)
                assert condition_encounters.issubset(encounter_ids)

    def test_patient_appears_at_multiple_facilities(self, sample_csvs, config):
        """Test that patients assigned to multiple facilities appear in each."""
        patients_df = sample_csvs["patients.csv"]
        encounters_df = sample_csvs["encounters.csv"]

        # Force all patients to 2 facilities
        config.facility_distribution.facility_count_weights = {1: 0.0, 2: 1.0}

        facility_assigner = FacilityAssigner(
            config.facility_distribution, random_seed=42
        )
        patient_facilities, encounter_facilities = (
            facility_assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )

        csv_splitter = CSVSplitter()
        facility_csvs = csv_splitter.split_csvs_by_facility(
            sample_csvs, patient_facilities, encounter_facilities
        )

        # Each patient should appear in exactly 2 facilities
        for patient_uuid, facilities in patient_facilities.items():
            assert len(facilities) == 2

            # Check patient appears in both facilities
            for facility_id in facilities:
                facility_patients = facility_csvs[facility_id]["patients.csv"]
                assert patient_uuid in facility_patients["Id"].values

    def test_uuid_preservation_across_facilities(self, sample_csvs, config):
        """Test that patient UUIDs are identical across all facilities."""
        patients_df = sample_csvs["patients.csv"]
        encounters_df = sample_csvs["encounters.csv"]

        facility_assigner = FacilityAssigner(
            config.facility_distribution, random_seed=42
        )
        patient_facilities, encounter_facilities = (
            facility_assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )

        csv_splitter = CSVSplitter()
        facility_csvs = csv_splitter.split_csvs_by_facility(
            sample_csvs, patient_facilities, encounter_facilities
        )

        # Apply errors
        error_injector = ErrorInjector(config.error_injection, random_seed=42)

        for facility_id, csvs in facility_csvs.items():
            errored_patients_df, _ = error_injector.inject_errors_into_patients(
                csvs["patients.csv"], facility_id
            )
            facility_csvs[facility_id]["patients.csv"] = errored_patients_df

        # For each patient at multiple facilities, verify UUID is identical
        for patient_uuid, facilities in patient_facilities.items():
            if len(facilities) < 2:
                continue

            # Get patient record from each facility
            facility_records = []
            for facility_id in facilities:
                facility_patients = facility_csvs[facility_id]["patients.csv"]
                patient_record = facility_patients[
                    facility_patients["Id"] == patient_uuid
                ]
                if len(patient_record) > 0:
                    facility_records.append(patient_record.iloc[0])

            # All records should have same UUID
            for record in facility_records:
                assert record["Id"] == patient_uuid

            # But demographic fields may differ (due to errors)
            if len(facility_records) >= 2:
                # At least check that it's possible for names to differ
                # (not guaranteed due to randomness, but structurally possible)
                pass  # Errors are random, so we can't assert they differ
