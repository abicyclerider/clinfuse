"""Unit tests for promote_model.py — champion/challenger promotion gate."""

import json
import os
import sys
import tempfile

import pytest
from mlflow.tracking import MlflowClient

# llm_classifier is not an installable package — add it to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from promote_model import MODEL_NAME, METRIC_KEY, promote


def _make_metrics_file(tmpdir: str, eval_f1: float) -> str:
    path = os.path.join(tmpdir, "train_metrics.json")
    with open(path, "w") as f:
        json.dump(
            {"eval_f1": eval_f1, "eval_precision": 0.95, "eval_recall": 0.94},
            f,
        )
    return path


def _make_history_db(tmpdir: str) -> str:
    """Create an empty MLflow-backed SQLite database."""
    db_path = os.path.join(tmpdir, "mlflow_history.db")
    # MLflow creates schema on first connection
    client = MlflowClient(tracking_uri=f"sqlite:///{db_path}")
    # Force schema creation by listing experiments
    client.search_experiments()
    return db_path


@pytest.mark.unit
class TestBootstrap:
    def test_first_model_promoted(self, tmp_path):
        """First training run is always promoted (no champion exists)."""
        db = _make_history_db(str(tmp_path))
        metrics = _make_metrics_file(str(tmp_path), eval_f1=0.90)
        output = os.path.join(str(tmp_path), "decision.json")

        decision = promote(db, metrics, output)

        assert decision["promoted"] is True
        assert decision["champion_f1"] is None
        assert decision["champion_version"] is None
        assert decision["model_version"] == 1

        # Verify champion alias was set
        client = MlflowClient(tracking_uri=f"sqlite:///{db}")
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        assert int(mv.version) == 1

    def test_decision_file_written(self, tmp_path):
        """The promotion decision JSON is written to disk."""
        db = _make_history_db(str(tmp_path))
        metrics = _make_metrics_file(str(tmp_path), eval_f1=0.90)
        output = os.path.join(str(tmp_path), "decision.json")

        promote(db, metrics, output)

        with open(output) as f:
            written = json.load(f)
        assert written["promoted"] is True
        assert written["challenger_f1"] == 0.90


@pytest.mark.unit
class TestChallengerWins:
    def test_higher_f1_promoted(self, tmp_path):
        """Challenger with higher eval_f1 replaces champion."""
        db = _make_history_db(str(tmp_path))
        output = os.path.join(str(tmp_path), "decision.json")

        # Bootstrap champion with f1=0.90
        metrics_v1 = _make_metrics_file(str(tmp_path), eval_f1=0.90)
        promote(db, metrics_v1, output)

        # Challenger with f1=0.95
        metrics_v2 = _make_metrics_file(str(tmp_path), eval_f1=0.95)
        decision = promote(db, metrics_v2, output)

        assert decision["promoted"] is True
        assert decision["challenger_f1"] == 0.95
        assert decision["champion_f1"] == 0.90
        assert decision["model_version"] == 2
        assert decision["champion_version"] == 1

        # Champion alias now points to v2
        client = MlflowClient(tracking_uri=f"sqlite:///{db}")
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        assert int(mv.version) == 2

    def test_equal_f1_promoted(self, tmp_path):
        """Challenger matching champion's eval_f1 is promoted (>= threshold)."""
        db = _make_history_db(str(tmp_path))
        output = os.path.join(str(tmp_path), "decision.json")

        metrics = _make_metrics_file(str(tmp_path), eval_f1=0.90)
        promote(db, metrics, output)

        decision = promote(db, metrics, output)

        assert decision["promoted"] is True
        assert decision["challenger_f1"] == 0.90
        assert decision["champion_f1"] == 0.90


@pytest.mark.unit
class TestChallengerLoses:
    def test_lower_f1_rejected(self, tmp_path):
        """Challenger with lower eval_f1 is rejected."""
        db = _make_history_db(str(tmp_path))
        output = os.path.join(str(tmp_path), "decision.json")

        # Bootstrap champion with f1=0.95
        metrics_v1 = _make_metrics_file(str(tmp_path), eval_f1=0.95)
        promote(db, metrics_v1, output)

        # Challenger with f1=0.90
        metrics_v2 = _make_metrics_file(str(tmp_path), eval_f1=0.90)
        decision = promote(db, metrics_v2, output)

        assert decision["promoted"] is False
        assert decision["challenger_f1"] == 0.90
        assert decision["champion_f1"] == 0.95
        assert decision["model_version"] == 2
        assert decision["champion_version"] == 1

        # Champion alias still on v1
        client = MlflowClient(tracking_uri=f"sqlite:///{db}")
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        assert int(mv.version) == 1

    def test_metrics_stored_on_rejected_version(self, tmp_path):
        """Even rejected versions have their metrics stored as tags."""
        db = _make_history_db(str(tmp_path))
        output = os.path.join(str(tmp_path), "decision.json")

        metrics_v1 = _make_metrics_file(str(tmp_path), eval_f1=0.95)
        promote(db, metrics_v1, output)

        metrics_v2 = _make_metrics_file(str(tmp_path), eval_f1=0.90)
        promote(db, metrics_v2, output)

        client = MlflowClient(tracking_uri=f"sqlite:///{db}")
        mv = client.get_model_version(MODEL_NAME, "2")
        assert float(mv.tags[METRIC_KEY]) == 0.90
