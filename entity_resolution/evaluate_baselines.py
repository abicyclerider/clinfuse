#!/usr/bin/env python3
"""
Baseline evaluation: Splink-only golden records at auto-match and best-F1 thresholds.

Produces output/baselines/baseline_comparison.json with three scenarios:
  - auto_match_only:  threshold at config auto_match_probability (0.99)
  - splink_best_f1:   sweep 0.01–0.99 for best pair-level F1
  - clinfuse:         read from golden_records evaluation_metrics.json
"""

import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml

from entity_resolution.core.evaluation import (
    evaluate_clusters,
    evaluate_golden_records,
)
from entity_resolution.core.golden_record import create_golden_records
from shared.data_loader import create_record_id, load_facility_patients
from shared.evaluation import calculate_confusion_matrix, calculate_metrics
from shared.ground_truth import (
    add_record_ids_to_ground_truth,
    generate_true_pairs_from_ground_truth,
    load_ground_truth,
)

logger = logging.getLogger(__name__)


def _build_matches_at_threshold(
    features_df: pd.DataFrame, threshold: float
) -> pd.Series:
    """Return a boolean Series of matched pairs at the given threshold."""
    above = features_df[features_df["match_probability"] >= threshold]
    pairs = [
        tuple(sorted([row["record_id_1"], row["record_id_2"]]))
        for _, row in above.iterrows()
    ]
    if not pairs:
        idx = pd.MultiIndex.from_tuples([], names=["record_id_1", "record_id_2"])
        return pd.Series([], dtype=bool, index=idx)
    idx = pd.MultiIndex.from_tuples(pairs, names=["record_id_1", "record_id_2"])
    return pd.Series(True, index=idx)


def _evaluate_scenario(
    features_df: pd.DataFrame,
    threshold: float,
    patients_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    true_pairs: set,
    cfg: dict,
) -> dict:
    """Build golden records at *threshold* and compute pair + cluster metrics."""
    matches = _build_matches_at_threshold(features_df, threshold)

    # Pair-level metrics
    pred_pairs = set(matches[matches].index.tolist())
    tp, fp, fn = calculate_confusion_matrix(pred_pairs, true_pairs)
    pair_metrics = calculate_metrics(tp, fp, fn)

    # Golden records + cluster metrics
    golden_df = create_golden_records(matches, patients_df, cfg)
    golden_metrics = evaluate_golden_records(golden_df, ground_truth_df)
    cluster_metrics = evaluate_clusters(golden_df, ground_truth_df)

    return {
        "pair_metrics": pair_metrics,
        "cluster_metrics": cluster_metrics,
        "golden_metrics": golden_metrics,
        "golden_df": golden_df,
    }


def _find_best_f1_threshold(features_df: pd.DataFrame, true_pairs: set) -> float:
    """Sweep thresholds to find the one maximizing pair-level F1."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1 = -1.0
    best_t = 0.5

    for t in thresholds:
        above = features_df[features_df["match_probability"] >= t]
        pred_set = {
            tuple(sorted([row["record_id_1"], row["record_id_2"]]))
            for _, row in above.iterrows()
        }
        tp, fp, fn = calculate_confusion_matrix(pred_set, true_pairs)
        m = calculate_metrics(tp, fp, fn)
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_t = float(t)

    return round(best_t, 2)


@click.command()
@click.option(
    "--augmented-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to augmented data directory",
)
@click.option(
    "--features",
    required=True,
    type=click.Path(exists=True),
    help="Path to features.parquet from resolve stage",
)
@click.option(
    "--clinfuse-metrics",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation_metrics.json from golden_records stage",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for baseline comparison",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to matching config YAML",
)
def main(augmented_dir, features, clinfuse_metrics, output_dir, config):
    """Evaluate Splink-only baselines and compare with ClinFuse."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(config) as f:
        cfg = yaml.safe_load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load patient data
    run_dir = Path(augmented_dir)
    patients_df = load_facility_patients(str(run_dir))
    patients_df = create_record_id(patients_df)
    logger.info(f"Loaded {len(patients_df)} patient records")

    # Load ground truth
    ground_truth_df = load_ground_truth(str(run_dir))
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)
    true_pairs = generate_true_pairs_from_ground_truth(ground_truth_df)
    logger.info(f"Ground truth: {len(true_pairs)} true pairs")

    # Load features
    features_df = pd.read_parquet(features)
    logger.info(f"Loaded {len(features_df)} feature vectors")

    # --- Scenario 1: Auto-match only ---
    auto_threshold = cfg["splink"]["auto_match_probability"]
    logger.info(f"Scenario: auto_match_only (threshold={auto_threshold})")
    auto_result = _evaluate_scenario(
        features_df, auto_threshold, patients_df, ground_truth_df, true_pairs, cfg
    )

    # --- Scenario 2: Splink best F1 ---
    best_threshold = _find_best_f1_threshold(features_df, true_pairs)
    logger.info(f"Scenario: splink_best_f1 (threshold={best_threshold})")
    best_result = _evaluate_scenario(
        features_df, best_threshold, patients_df, ground_truth_df, true_pairs, cfg
    )

    # Save best-F1 golden records for inspection
    best_result["golden_df"].to_parquet(
        out / "splink_best_f1_golden_records.parquet", index=False
    )

    # --- Scenario 3: ClinFuse (read pre-computed) ---
    with open(clinfuse_metrics) as f:
        cf_metrics = json.load(f)

    # Extract pair and cluster metrics from the combined dict
    pair_keys = [
        "precision",
        "recall",
        "f1_score",
        "true_positives",
        "false_positives",
        "false_negatives",
    ]
    cluster_keys = [
        "n_clusters",
        "purity",
        "completeness",
        "split_patients",
        "impure_clusters",
        "perfect_resolution_rate",
    ]

    cf_pair = {k: cf_metrics[k] for k in pair_keys if k in cf_metrics}
    cf_cluster = {k: cf_metrics[k] for k in cluster_keys if k in cf_metrics}

    # --- Build comparison JSON ---
    comparison = {
        "scenarios": {
            "auto_match_only": {
                "threshold": auto_threshold,
                "pair_metrics": auto_result["pair_metrics"],
                "cluster_metrics": auto_result["cluster_metrics"],
            },
            "splink_best_f1": {
                "threshold": best_threshold,
                "pair_metrics": best_result["pair_metrics"],
                "cluster_metrics": best_result["cluster_metrics"],
            },
            "clinfuse": {
                "pair_metrics": cf_pair,
                "cluster_metrics": cf_cluster,
            },
        }
    }

    with open(out / "baseline_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info(f"Saved baseline_comparison.json to {out}")

    # Summary
    for name, scenario in comparison["scenarios"].items():
        pm = scenario["pair_metrics"]
        cm = scenario.get("cluster_metrics", {})
        logger.info(
            f"  {name}: F1={pm.get('f1_score', 'N/A'):.3f}, "
            f"purity={cm.get('purity', 'N/A')}, "
            f"completeness={cm.get('completeness', 'N/A')}, "
            f"split={cm.get('split_patients', 'N/A')}"
        )


if __name__ == "__main__":
    main()
