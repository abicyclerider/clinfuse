"""
Unit tests for Splink linker module.
"""

import math

import pandas as pd
import pytest

from entity_resolution.core.splink_linker import (
    build_settings,
    classify_predictions,
    create_linker,
    evaluate_splink_only,
    predict_matches,
    splink_logit,
    train_model,
)


@pytest.fixture
def sample_config():
    """Sample configuration for Splink."""
    return {
        "splink": {
            "predict_threshold": 0.01,
            "auto_match_probability": 0.95,
            "auto_reject_probability": 0.05,
        },
        "gray_zone": {
            "w_splink": 0.5,
            "w_llm": 1.0,
            "threshold": 0.0,
        },
    }


@pytest.fixture
def sample_patients():
    """Sample patient data for Splink testing.

    Creates two pairs of matching patients across facilities.
    """
    data = {
        "record_id": [
            "fac1_p1",
            "fac2_p1",  # Match pair 1
            "fac1_p2",
            "fac2_p2",  # Match pair 2
            "fac1_p3",  # Non-match
        ],
        "first_name": ["John", "John", "Jane", "Jane", "Bob"],
        "last_name": ["Smith", "Smith", "Doe", "Doe", "Jones"],
        "address": [
            "123 Main St",
            "123 Main St",
            "456 Oak Ave",
            "456 Oak Ave",
            "789 Pine Rd",
        ],
        "city": ["Boston", "Boston", "Boston", "Boston", "Springfield"],
        "state": ["MA", "MA", "MA", "MA", "MA"],
        "zip": ["02115", "02115", "02116", "02116", "01101"],
        "ssn": [
            "111-22-3333",
            "111-22-3333",
            "444-55-6666",
            "444-55-6666",
            "777-88-9999",
        ],
        "birthdate": [
            "1980-01-15",
            "1980-01-15",
            "1990-06-20",
            "1990-06-20",
            "1975-03-10",
        ],
        "birth_year": [1980, 1980, 1990, 1990, 1975],
    }
    return pd.DataFrame(data)


def test_build_settings(sample_config):
    """Test that build_settings returns valid SettingsCreator."""
    settings, threshold = build_settings(sample_config)
    assert settings is not None
    assert threshold == 0.01


def test_create_linker(sample_patients, sample_config):
    """Test that Linker can be created with sample data."""
    linker, threshold = create_linker(sample_patients, sample_config)
    assert linker is not None
    assert threshold == 0.01


def test_train_and_predict(sample_patients, sample_config):
    """Test full train + predict pipeline with sample data."""
    linker, _ = create_linker(sample_patients, sample_config)
    train_model(linker)
    predictions = predict_matches(linker, sample_config)

    assert isinstance(predictions, pd.DataFrame)
    assert "match_probability" in predictions.columns
    assert "match_weight" in predictions.columns
    assert len(predictions) > 0


def test_classify_predictions(sample_config):
    """Test classification of predictions into tiers."""
    predictions = pd.DataFrame(
        {
            "record_id_l": ["fac1_p1", "fac1_p2", "fac1_p3"],
            "record_id_r": ["fac2_p1", "fac2_p2", "fac2_p2"],
            "match_probability": [0.99, 0.50, 0.01],
            "match_weight": [10.0, 0.0, -5.0],
        }
    )

    auto, gray, all_pred = classify_predictions(predictions, sample_config)

    # 0.99 >= 0.95 -> auto_match
    assert len(auto) == 1
    assert auto.iloc[0]["record_id_1"] == "fac1_p1"

    # 0.50 is between 0.05 and 0.95 -> gray zone
    assert len(gray) == 1

    # All predictions should have total_score column
    assert "total_score" in all_pred.columns
    assert "record_id_1" in all_pred.columns


def test_evaluate_splink_only(sample_config):
    """Test Splink-only evaluation metrics."""
    predictions = pd.DataFrame(
        {
            "record_id_1": ["fac1_p1", "fac1_p2"],
            "record_id_2": ["fac2_p1", "fac2_p2"],
            "match_probability": [0.99, 0.80],
        }
    )
    true_pairs = {
        ("fac1_p1", "fac2_p1"),
        ("fac1_p2", "fac2_p2"),
    }

    metrics = evaluate_splink_only(predictions, true_pairs, sample_config)

    assert "splink_only_auto_f1" in metrics
    assert "splink_only_best_f1" in metrics
    assert "splink_only_best_threshold" in metrics
    # Both are true matches, so at threshold 0.80 we'd get perfect F1
    assert metrics["splink_only_best_f1"] == 1.0


def test_splink_logit():
    """Test logit conversion."""
    # 0.5 -> 0.0 log-odds
    assert abs(splink_logit(0.5)) < 1e-6
    # 0.95 -> positive
    assert splink_logit(0.95) > 0
    # 0.05 -> negative
    assert splink_logit(0.05) < 0
    # Clipping at extremes
    assert math.isfinite(splink_logit(0.0))
    assert math.isfinite(splink_logit(1.0))
