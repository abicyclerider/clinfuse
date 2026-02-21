"""Unit tests for evaluate_clusters() in entity_resolution/core/evaluation.py."""

import pandas as pd
import pytest

from entity_resolution.core.evaluation import evaluate_clusters


def _make_golden(clusters: list[list[str]]) -> pd.DataFrame:
    """Build a minimal golden_records DataFrame from a list of record-ID lists."""
    rows = []
    for i, rids in enumerate(clusters):
        rows.append(
            {
                "golden_id": f"g{i}",
                "source_record_ids": ",".join(rids),
                "num_records": len(rids),
            }
        )
    return pd.DataFrame(rows)


def _make_gt(mapping: dict[str, str]) -> pd.DataFrame:
    """Build a ground_truth DataFrame from {record_id: true_patient_id}."""
    rows = [{"record_id": rid, "true_patient_id": pid} for rid, pid in mapping.items()]
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestEvaluateClusters:
    def test_perfect_resolution(self):
        """All records perfectly clustered by true patient."""
        golden = _make_golden([["r1", "r2"], ["r3", "r4"]])
        gt = _make_gt({"r1": "A", "r2": "A", "r3": "B", "r4": "B"})

        m = evaluate_clusters(golden, gt)

        assert m["n_clusters"] == 2
        assert m["purity"] == 1.0
        assert m["completeness"] == 1.0
        assert m["split_patients"] == 0
        assert m["impure_clusters"] == 0
        assert m["perfect_resolution_rate"] == 1.0

    def test_impure_cluster(self):
        """One cluster mixes records from two true patients."""
        golden = _make_golden([["r1", "r2", "r3"]])
        gt = _make_gt({"r1": "A", "r2": "A", "r3": "B"})

        m = evaluate_clusters(golden, gt)

        assert m["impure_clusters"] == 1
        assert m["purity"] == 0.0  # no pure clusters at all
        # Both A and B are complete (all records in one cluster)
        assert m["completeness"] == 1.0
        # Neither is "perfect" — the cluster is impure
        assert m["perfect_resolution_rate"] == 0.0

    def test_split_patient(self):
        """Patient A's records split across two clusters."""
        golden = _make_golden([["r1"], ["r2"]])
        gt = _make_gt({"r1": "A", "r2": "A"})

        m = evaluate_clusters(golden, gt)

        assert m["split_patients"] == 1
        assert m["completeness"] == 0.0
        assert m["purity"] == 1.0  # each cluster is pure (single record)
        assert m["perfect_resolution_rate"] == 0.0

    def test_singletons_only(self):
        """Every record in its own cluster — all pure, all split."""
        golden = _make_golden([["r1"], ["r2"], ["r3"], ["r4"]])
        gt = _make_gt({"r1": "A", "r2": "A", "r3": "B", "r4": "B"})

        m = evaluate_clusters(golden, gt)

        assert m["n_clusters"] == 4
        assert m["purity"] == 1.0
        assert m["completeness"] == 0.0
        assert m["split_patients"] == 2
        assert m["perfect_resolution_rate"] == 0.0

    def test_mixed_scenario(self):
        """Patient A perfect, patient B split, patient C impure merge with D."""
        golden = _make_golden(
            [
                ["r1", "r2"],  # A: pure and complete
                ["r3"],  # B fragment 1
                ["r4"],  # B fragment 2
                ["r5", "r6"],  # C + D mixed (impure)
            ]
        )
        gt = _make_gt(
            {
                "r1": "A",
                "r2": "A",
                "r3": "B",
                "r4": "B",
                "r5": "C",
                "r6": "D",
            }
        )

        m = evaluate_clusters(golden, gt)

        assert m["n_clusters"] == 4
        assert m["impure_clusters"] == 1
        assert m["split_patients"] == 1  # B
        # Complete: A (1 cluster), C (1 cluster), D (1 cluster) = 3 of 4
        assert m["completeness"] == 0.75
        # Pure records: r1+r2 (2) + r3 (1) + r4 (1) = 4 of 6 total
        assert m["purity"] == pytest.approx(4 / 6, abs=0.001)
        # Multi-record patients: A (2 records), B (2 records)
        # A is perfect (complete + pure), B is not (split)
        assert m["perfect_resolution_rate"] == 0.5

    def test_singleton_true_patients(self):
        """True patients with only 1 record are NOT multi-record."""
        golden = _make_golden([["r1"], ["r2"]])
        gt = _make_gt({"r1": "A", "r2": "B"})

        m = evaluate_clusters(golden, gt)

        assert m["n_clusters"] == 2
        assert m["purity"] == 1.0
        assert m["completeness"] == 1.0
        assert m["split_patients"] == 0
        # No multi-record patients → rate is 0.0 (vacuously)
        assert m["perfect_resolution_rate"] == 0.0
