"""
Evaluation metrics for entity resolution performance.

Compares predicted matches against ground truth to calculate precision, recall, F1.
Delegates core metrics to shared.evaluation.
"""

import logging

import pandas as pd

from shared.evaluation import (
    calculate_confusion_matrix,
    calculate_metrics,
)
from shared.ground_truth import generate_true_pairs_from_ground_truth

logger = logging.getLogger(__name__)


def evaluate_matches(
    predicted_matches: pd.Series,
    ground_truth: pd.DataFrame,
    patient_data: pd.DataFrame = None,
) -> dict:
    """
    Evaluate predicted matches against ground truth.

    Args:
        predicted_matches: Boolean Series with MultiIndex (record_id_1, record_id_2)
        ground_truth: Ground truth DataFrame with true_patient_id
        patient_data: Patient DataFrame with record_id (optional)

    Returns:
        Dictionary with evaluation metrics
    """
    predicted_pairs = set(predicted_matches[predicted_matches].index.tolist())
    true_pairs = generate_true_pairs_from_ground_truth(ground_truth)

    tp, fp, fn = calculate_confusion_matrix(predicted_pairs, true_pairs)
    metrics = calculate_metrics(tp, fp, fn)

    metrics["predicted_pairs"] = len(predicted_pairs)
    metrics["true_pairs"] = len(true_pairs)

    logger.info(
        f"Evaluation: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}"
    )

    return metrics


def evaluate_golden_records(
    golden_records: pd.DataFrame, ground_truth: pd.DataFrame
) -> dict:
    """
    Evaluate quality of golden records against ground truth.

    Args:
        golden_records: DataFrame with golden records
        ground_truth: Ground truth DataFrame

    Returns:
        Dictionary with golden record metrics
    """
    num_golden = len(golden_records)
    true_id_col = (
        "true_patient_id"
        if "true_patient_id" in ground_truth.columns
        else "original_patient_uuid"
    )
    num_true_patients = ground_truth[true_id_col].nunique()

    count_accuracy = num_golden == num_true_patients

    metrics = {
        "num_golden_records": num_golden,
        "num_true_patients": num_true_patients,
        "count_accuracy": count_accuracy,
        "count_difference": num_golden - num_true_patients,
    }

    if "num_records" in golden_records.columns:
        metrics["avg_records_per_patient"] = golden_records["num_records"].mean()
        metrics["max_records_per_patient"] = golden_records["num_records"].max()

    logger.info(f"Golden records: {num_golden} (expected {num_true_patients})")

    return metrics


def evaluate_clusters(
    golden_records: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    """
    Evaluate cluster-level quality of golden records against ground truth.

    Measures how well the clustering recovers true patient groupings:
    purity (are clusters homogeneous?), completeness (are patients' records
    gathered into one cluster?), and perfect resolution rate.

    Args:
        golden_records: DataFrame with ``source_record_ids`` column
            (comma-separated record IDs per cluster).
        ground_truth: DataFrame with ``record_id`` and either
            ``true_patient_id`` or ``original_patient_uuid``.

    Returns:
        Dictionary with cluster-level metrics.
    """
    true_id_col = (
        "true_patient_id"
        if "true_patient_id" in ground_truth.columns
        else "original_patient_uuid"
    )

    # record_id → true_patient_id lookup
    rid_to_true = dict(zip(ground_truth["record_id"], ground_truth[true_id_col]))

    # Parse clusters
    clusters: list[list[str]] = []
    for _, row in golden_records.iterrows():
        src = row.get("source_record_ids", "")
        if pd.isna(src) or src == "":
            continue
        clusters.append([rid.strip() for rid in str(src).split(",")])

    n_clusters = len(clusters)

    # Per-cluster: which true patients appear, and how many records per patient
    impure_clusters = 0
    weighted_pure_records = 0
    total_records = 0

    # Per-true-patient: track which clusters contain their records
    from collections import defaultdict

    patient_to_clusters: dict[str, set[int]] = defaultdict(set)
    patient_record_counts: dict[str, int] = defaultdict(int)

    for cluster_idx, record_ids in enumerate(clusters):
        patient_counts: dict[str, int] = defaultdict(int)
        for rid in record_ids:
            true_pid = rid_to_true.get(rid)
            if true_pid is not None:
                patient_counts[true_pid] += 1
                patient_to_clusters[true_pid].add(cluster_idx)
                patient_record_counts[true_pid] += 1

        if not patient_counts:
            continue

        n_records_in_cluster = sum(patient_counts.values())
        total_records += n_records_in_cluster

        # Cluster is pure if all records belong to the same true patient
        if len(patient_counts) > 1:
            impure_clusters += 1
        else:
            weighted_pure_records += n_records_in_cluster

    # Purity: weighted fraction of records in pure clusters
    purity = weighted_pure_records / total_records if total_records > 0 else 0.0

    # Completeness & split patients
    split_patients = 0
    complete_patients = 0
    n_true_patients = len(patient_to_clusters)

    for pid, cluster_set in patient_to_clusters.items():
        if len(cluster_set) == 1:
            complete_patients += 1
        else:
            split_patients += 1

    completeness = complete_patients / n_true_patients if n_true_patients > 0 else 0.0

    # Perfect resolution rate: fraction of multi-record patients that are
    # both complete (all records in one cluster) AND pure (that cluster
    # contains only their records).
    multi_record_patients = {
        pid for pid, cnt in patient_record_counts.items() if cnt > 1
    }
    n_multi = len(multi_record_patients)

    # Build cluster_idx → set of true patients for purity check
    cluster_patients: dict[int, set[str]] = defaultdict(set)
    for cluster_idx, record_ids in enumerate(clusters):
        for rid in record_ids:
            true_pid = rid_to_true.get(rid)
            if true_pid is not None:
                cluster_patients[cluster_idx].add(true_pid)

    perfect = 0
    for pid in multi_record_patients:
        c_set = patient_to_clusters[pid]
        if len(c_set) == 1:
            # Complete — now check if that cluster is pure
            (cluster_idx,) = c_set
            if len(cluster_patients[cluster_idx]) == 1:
                perfect += 1

    perfect_resolution_rate = perfect / n_multi if n_multi > 0 else 0.0

    metrics = {
        "n_clusters": n_clusters,
        "purity": round(purity, 4),
        "completeness": round(completeness, 4),
        "split_patients": split_patients,
        "impure_clusters": impure_clusters,
        "perfect_resolution_rate": round(perfect_resolution_rate, 4),
    }

    logger.info(
        f"Cluster evaluation: purity={metrics['purity']:.3f}, "
        f"completeness={metrics['completeness']:.3f}, "
        f"split={split_patients}, perfect_rate={metrics['perfect_resolution_rate']:.3f}"
    )

    return metrics
