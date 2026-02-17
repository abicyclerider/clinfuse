"""Unit tests for augmentation/core/data_splitter.py — referential integrity logic."""

import pandas as pd
import pytest

from augmentation.core.data_splitter import DataSplitter


@pytest.fixture
def splitter():
    return DataSplitter()


@pytest.fixture
def minimal_data():
    """Minimal dataset: 3 patients, 6 encounters, 2 facilities."""
    patients = pd.DataFrame(
        {
            "Id": ["p1", "p2", "p3"],
            "FIRST": ["Alice", "Bob", "Charlie"],
            "LAST": ["A", "B", "C"],
        }
    )
    encounters = pd.DataFrame(
        {
            "Id": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "PATIENT": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "START": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-06-01",
                    "2020-02-01",
                    "2020-07-01",
                    "2020-03-01",
                    "2020-08-01",
                ]
            ),
        }
    )
    conditions = pd.DataFrame(
        {
            "ENCOUNTER": ["e1", "e3", "e5"],
            "DESCRIPTION": ["Flu", "Cold", "Fever"],
        }
    )
    medications = pd.DataFrame(
        {
            "ENCOUNTER": ["e2", "e4", "e6"],
            "DESCRIPTION": ["Aspirin", "Ibuprofen", "Tylenol"],
        }
    )
    organizations = pd.DataFrame(
        {
            "Id": ["org1", "org2"],
            "NAME": ["Hospital A", "Hospital B"],
        }
    )
    claims = pd.DataFrame(
        {
            "Id": ["c1", "c2", "c3"],
            "APPOINTMENTID": ["e1", "e3", "e5"],
        }
    )
    # Patient-facility mapping: p1 at both, p2 at fac 1 only, p3 at fac 2 only
    patient_facilities = {
        "p1": [1, 2],
        "p2": [1],
        "p3": [2],
    }
    # Encounter-facility mapping
    encounter_facilities = {
        "e1": 1,
        "e2": 2,  # p1's second encounter at facility 2
        "e3": 1,
        "e4": 1,
        "e5": 2,
        "e6": 2,
    }
    synthea_tables = {
        "patients": patients,
        "encounters": encounters,
        "conditions": conditions,
        "medications": medications,
        "organizations": organizations,
        "claims": claims,
    }
    return synthea_tables, patient_facilities, encounter_facilities


@pytest.mark.unit
class TestDataSplitter:
    def test_patients_filtered_to_facility(self, splitter, minimal_data):
        """Only patients with encounters at facility appear in that facility's data."""
        tables, pat_fac, enc_fac = minimal_data
        result = splitter.split_tables_by_facility(tables, pat_fac, enc_fac)

        # Facility 1: p1 (both facilities) and p2 (fac 1 only)
        fac1_patients = set(result[1]["patients"]["Id"])
        assert fac1_patients == {"p1", "p2"}

        # Facility 2: p1 (both facilities) and p3 (fac 2 only)
        fac2_patients = set(result[2]["patients"]["Id"])
        assert fac2_patients == {"p1", "p3"}

    def test_encounters_filtered_to_facility(self, splitter, minimal_data):
        """Only encounters assigned to a facility appear there."""
        tables, pat_fac, enc_fac = minimal_data
        result = splitter.split_tables_by_facility(tables, pat_fac, enc_fac)

        fac1_encounters = set(result[1]["encounters"]["Id"])
        assert fac1_encounters == {"e1", "e3", "e4"}

        fac2_encounters = set(result[2]["encounters"]["Id"])
        assert fac2_encounters == {"e2", "e5", "e6"}

    def test_encounter_linked_tables_follow_encounters(self, splitter, minimal_data):
        """Conditions/medications follow their encounter's facility assignment."""
        tables, pat_fac, enc_fac = minimal_data
        result = splitter.split_tables_by_facility(tables, pat_fac, enc_fac)

        # Conditions: e1 at fac1, e5 at fac2, e3 at fac1
        fac1_conditions = set(result[1]["conditions"]["ENCOUNTER"])
        assert fac1_conditions == {"e1", "e3"}

        fac2_conditions = set(result[2]["conditions"]["ENCOUNTER"])
        assert fac2_conditions == {"e5"}

    def test_reference_tables_copied_to_all(self, splitter, minimal_data):
        """Reference tables (organizations, providers, payers) appear in every facility."""
        tables, pat_fac, enc_fac = minimal_data
        result = splitter.split_tables_by_facility(tables, pat_fac, enc_fac)

        for fac_id in [1, 2]:
            assert len(result[fac_id]["organizations"]) == 2

    def test_claims_follow_encounters(self, splitter, minimal_data):
        """Claims are linked via APPOINTMENTID (encounter)."""
        tables, pat_fac, enc_fac = minimal_data
        result = splitter.split_tables_by_facility(tables, pat_fac, enc_fac)

        # Claim c1 -> e1 (fac1), c2 -> e3 (fac1), c3 -> e5 (fac2)
        fac1_claims = set(result[1]["claims"]["APPOINTMENTID"])
        assert fac1_claims == {"e1", "e3"}

        fac2_claims = set(result[2]["claims"]["APPOINTMENTID"])
        assert fac2_claims == {"e5"}
