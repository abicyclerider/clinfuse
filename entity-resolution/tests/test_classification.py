"""
Unit tests for classification module.
"""

import pandas as pd
import pytest
from src.classification import apply_deterministic_rules, classify_threshold


@pytest.fixture
def sample_features():
    """Sample features for testing."""
    return pd.DataFrame(
        {
            "first_name_sim": [0.9, 0.7, 0.95],
            "last_name_sim": [1.0, 0.8, 1.0],
            "ssn_match": [1, 0, 1],
            "birthdate_match": [1, 0, 1],
            "total_score": [4.5, 2.5, 5.0],
        }
    )


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {"classification": {"method": "threshold", "threshold": 3.5}}


def test_classify_threshold(sample_features, sample_config):
    """Test threshold-based classification."""
    matches = classify_threshold(sample_features, sample_config)

    # Should match records with total_score >= 3.5
    assert matches[0] == True  # score 4.5
    assert matches[1] == False  # score 2.5
    assert matches[2] == True  # score 5.0


def test_apply_deterministic_rules(sample_features):
    """Test deterministic matching rules."""
    matches = apply_deterministic_rules(sample_features)

    # Should match records with SSN and birthdate exact match
    assert matches[0] == True  # SSN=1, DOB=1
    assert matches[1] == False  # SSN=0, DOB=0
    assert matches[2] == True  # SSN=1, DOB=1
