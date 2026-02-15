"""
Unit tests for comparison module.
"""

import pandas as pd
import pytest
from src.comparison import (
    add_composite_features,
    create_custom_comparator,
)


@pytest.fixture
def sample_patients():
    """Sample patient data for testing."""
    data = {
        "record_id": ["fac1_1", "fac2_1"],
        "first_name": ["John", "Jon"],
        "last_name": ["Smith", "Smith"],
        "maiden_name": [None, None],
        "address": ["123 Main St", "123 Main Street"],
        "city": ["Boston", "Boston"],
        "state": ["MA", "MA"],
        "zip": ["02115", "02115"],
        "ssn": ["123456789", "123456789"],
        "birthdate": pd.to_datetime(["1980-01-01", "1980-01-01"]),
        "gender": ["M", "M"],
    }
    return pd.DataFrame(data).set_index("record_id")


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "comparison": {
            "first_name_threshold": 0.85,
            "last_name_threshold": 0.85,
            "address_threshold": 0.80,
            "ssn_exact_match": True,
            "birthdate_tolerance_days": 1,
        }
    }


def test_create_custom_comparator(sample_config):
    """Test creation of custom comparator."""
    comparator = create_custom_comparator(sample_config["comparison"])

    # Should have comparison methods defined
    assert comparator is not None


def test_add_composite_features():
    """Test addition of composite features."""
    features = pd.DataFrame(
        {
            "first_name_sim": [0.9, 0.7],
            "last_name_sim": [1.0, 0.8],
            "address_sim": [0.85, 0.6],
        }
    )

    result = add_composite_features(features)

    assert "total_score" in result.columns
    assert result["total_score"][0] == 0.9 + 1.0 + 0.85
