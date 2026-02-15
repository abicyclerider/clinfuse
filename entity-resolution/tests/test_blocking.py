"""
Unit tests for blocking module.
"""

import pandas as pd
import pytest
from src.blocking import (
    analyze_blocking_statistics,
    create_candidate_pairs,
    generate_true_pairs_from_ground_truth,
)


@pytest.fixture
def sample_patients():
    """Sample patient data for testing."""
    data = {
        "record_id": ["fac1_1", "fac1_2", "fac2_1", "fac2_2"],
        "first_name": ["John", "Jane", "John", "Bob"],
        "last_name": ["Smith", "Doe", "Smith", "Jones"],
        "state": ["MA", "MA", "MA", "NY"],
        "zip": ["02115", "02116", "02115", "10001"],
        "birthdate": pd.to_datetime(
            ["1980-01-01", "1990-02-15", "1980-01-01", "1975-05-20"]
        ),
    }
    df = pd.DataFrame(data)
    df["birth_year"] = df["birthdate"].dt.year
    return df.set_index("record_id")


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth for testing."""
    return pd.DataFrame(
        {
            "facility_id": ["fac1", "fac2", "fac1", "fac2"],
            "patient_id": ["1", "1", "2", "2"],
            "true_patient_id": [
                "patient_001",
                "patient_001",
                "patient_002",
                "patient_002",
            ],
            "record_id": ["fac1_1", "fac2_1", "fac1_2", "fac2_2"],
        }
    )


def test_create_candidate_pairs_lastname_state(sample_patients):
    """Test blocking on last name and state."""
    pairs = create_candidate_pairs(sample_patients, strategy="lastname_state")

    # Should generate pairs for records with same last name and state
    assert len(pairs) > 0


def test_generate_true_pairs(sample_ground_truth):
    """Test generation of true matching pairs from ground truth."""
    true_pairs = generate_true_pairs_from_ground_truth(sample_ground_truth)

    # Should have 2 true pairs (fac1_1, fac2_1) and (fac1_2, fac2_2)
    assert len(true_pairs) == 2

    # Check pairs are normalized (sorted)
    for pair in true_pairs:
        assert pair[0] <= pair[1]


def test_analyze_blocking_statistics(sample_patients):
    """Test blocking statistics analysis."""
    stats = analyze_blocking_statistics(sample_patients, "last_name")

    assert stats["field"] == "last_name"
    assert stats["total_records"] == 4
    assert stats["unique_values"] == 3  # Smith, Doe, Jones
    assert 0 < stats["cardinality_ratio"] <= 1
