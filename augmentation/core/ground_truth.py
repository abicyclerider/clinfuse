"""Ground truth tracking and output generation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


class GroundTruthTracker:
    """Tracks patient-facility mappings and error transformations."""

    def __init__(self):
        """Initialize ground truth tracker."""
        self.patient_facility_records = []
        self.error_log = []

    def add_patient_facility_mapping(
        self,
        patient_uuid: str,
        facility_id: int,
        num_encounters: int,
        patient_data: Dict,
        errors_applied: List[str],
    ) -> None:
        """
        Record a patient's presence at a facility.

        Args:
            patient_uuid: Patient UUID
            facility_id: Facility identifier
            num_encounters: Number of encounters at this facility
            patient_data: Patient demographic data (possibly with errors)
            errors_applied: List of error types applied
        """
        record = {
            "original_patient_uuid": patient_uuid,
            "facility_id": facility_id,
            "num_encounters": num_encounters,
            "errors_applied": ",".join(errors_applied) if errors_applied else "none",
            "first_name": patient_data.get("FIRST"),
            "last_name": patient_data.get("LAST"),
            "maiden_name": patient_data.get("MAIDEN"),
            "ssn": patient_data.get("SSN"),
            "birthdate": patient_data.get("BIRTHDATE"),
            "gender": patient_data.get("GENDER"),
            "address": patient_data.get("ADDRESS"),
            "city": patient_data.get("CITY"),
            "zip": patient_data.get("ZIP"),
        }

        self.patient_facility_records.append(record)

    def add_error_records(self, error_records: List[Dict]) -> None:
        """
        Add error transformation records.

        Args:
            error_records: List of error records from ErrorInjector
        """
        # Add timestamp to each record
        for record in error_records:
            record["timestamp"] = datetime.now().isoformat()

        self.error_log.extend(error_records)

    def export_ground_truth_csv(self, output_path: Path) -> None:
        """
        Export ground truth CSV file.

        Args:
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.patient_facility_records)

        # Sort by patient UUID and facility ID
        df = df.sort_values(["original_patient_uuid", "facility_id"])

        df.to_csv(output_path, index=False)

    def export_error_log_jsonl(self, output_path: Path) -> None:
        """
        Export detailed error log as JSONL.

        Args:
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for record in self.error_log:
                f.write(json.dumps(record) + "\n")

    def generate_statistics(self) -> Dict:
        """
        Generate statistics about ground truth data.

        Returns:
            Dictionary with statistics
        """
        if not self.patient_facility_records:
            return {
                "total_patient_facility_pairs": 0,
                "unique_patients": 0,
                "patients_with_errors": 0,
                "error_rate": 0.0,
            }

        df = pd.DataFrame(self.patient_facility_records)

        # Count unique patients
        unique_patients = df["original_patient_uuid"].nunique()

        # Count patients with errors
        patients_with_errors = df[df["errors_applied"] != "none"][
            "original_patient_uuid"
        ].nunique()

        # Calculate error rate
        error_rate = (
            patients_with_errors / unique_patients if unique_patients > 0 else 0.0
        )

        # Count patients by number of facilities
        patients_per_facility_count = df.groupby("original_patient_uuid")[
            "facility_id"
        ].count()
        facility_count_distribution = (
            patients_per_facility_count.value_counts().to_dict()
        )

        return {
            "total_patient_facility_pairs": len(self.patient_facility_records),
            "unique_patients": unique_patients,
            "patients_with_errors": patients_with_errors,
            "error_rate": error_rate,
            "facility_count_distribution": facility_count_distribution,
            "total_error_transformations": len(self.error_log),
        }

    def export_statistics_json(
        self, output_path: Path, additional_stats: Dict = None
    ) -> None:
        """
        Export statistics as JSON.

        Args:
            output_path: Output file path
            additional_stats: Optional additional statistics to include
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.generate_statistics()

        if additional_stats:
            stats.update(additional_stats)

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
