"""
Evaluation metrics for entity resolution performance.

Compares predicted matches against ground truth to calculate precision, recall, F1.
Delegates core metrics to shared.evaluation.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so shared module is importable
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.evaluation import (  # noqa: E402
    calculate_confusion_matrix,
    calculate_metrics,
)

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


def generate_true_pairs_from_ground_truth(ground_truth: pd.DataFrame) -> set:
    """
    Generate set of true matching pairs from ground truth.

    Args:
        ground_truth: DataFrame with facility_id, patient_id, true_patient_id, record_id

    Returns:
        Set of tuples (record_id_1, record_id_2) for true matches
    """
    true_id_col = (
        "true_patient_id"
        if "true_patient_id" in ground_truth.columns
        else "original_patient_uuid"
    )

    true_pairs = set()

    for true_id, group in ground_truth.groupby(true_id_col):
        record_ids = (
            group["record_id"].dropna().tolist() if "record_id" in group.columns else []
        )

        for i in range(len(record_ids)):
            for j in range(i + 1, len(record_ids)):
                pair = tuple(sorted([record_ids[i], record_ids[j]]))
                true_pairs.add(pair)

    return true_pairs


def error_analysis(
    predicted_matches: pd.Series,
    ground_truth: pd.DataFrame,
    features: pd.DataFrame,
    patient_data: pd.DataFrame,
) -> dict:
    """
    Analyze false positives and false negatives to understand errors.

    Args:
        predicted_matches: Boolean Series with predictions
        ground_truth: Ground truth DataFrame
        features: DataFrame with similarity scores
        patient_data: Patient DataFrame with record details

    Returns:
        Dictionary with error analysis
    """
    predicted_pairs = set(predicted_matches[predicted_matches].index.tolist())
    true_pairs = generate_true_pairs_from_ground_truth(ground_truth)

    predicted_normalized = {tuple(sorted(p)) for p in predicted_pairs}
    true_normalized = {tuple(sorted(p)) for p in true_pairs}

    false_positives = predicted_normalized - true_normalized
    false_negatives = true_normalized - predicted_normalized

    logger.info(f"Error analysis: {len(false_positives)} FP, {len(false_negatives)} FN")

    fp_analysis = analyze_error_pairs(
        false_positives, features, patient_data, "false_positive"
    )
    fn_analysis = analyze_error_pairs(
        false_negatives, features, patient_data, "false_negative"
    )

    return {
        "false_positives": fp_analysis,
        "false_negatives": fn_analysis,
        "fp_count": len(false_positives),
        "fn_count": len(false_negatives),
    }


def analyze_error_pairs(
    error_pairs: set,
    features: pd.DataFrame,
    patient_data: pd.DataFrame,
    error_type: str,
) -> pd.DataFrame:
    """
    Analyze characteristics of error pairs.

    Args:
        error_pairs: Set of error pairs
        features: DataFrame with similarity scores
        patient_data: Patient DataFrame
        error_type: 'false_positive' or 'false_negative'

    Returns:
        DataFrame with error analysis
    """
    error_data = []

    for pair in error_pairs:
        if pair in features.index or (pair[1], pair[0]) in features.index:
            if pair in features.index:
                feat_row = features.loc[pair]
            else:
                feat_row = features.loc[(pair[1], pair[0])]

            error_data.append(
                {
                    "record_id_1": pair[0],
                    "record_id_2": pair[1],
                    "error_type": error_type,
                    "total_score": feat_row.get("total_score", feat_row.sum()),
                    "name_score": feat_row.get("name_score", None),
                    "address_score": feat_row.get("address_score", None),
                }
            )

    if not error_data:
        return pd.DataFrame()

    error_df = pd.DataFrame(error_data)

    logger.debug(f"{error_type} avg score: {error_df['total_score'].mean():.2f}")

    return error_df


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


def calculate_cluster_purity(clusters: list, ground_truth: pd.DataFrame) -> float:
    """
    Calculate purity of clusters (golden records).

    Args:
        clusters: List of sets containing record_ids
        ground_truth: Ground truth DataFrame

    Returns:
        Average purity score (0-1)
    """
    record_to_true = {}
    if "record_id" in ground_truth.columns:
        record_to_true = dict(
            zip(ground_truth["record_id"], ground_truth["true_patient_id"])
        )

    purities = []

    for cluster in clusters:
        true_ids = [record_to_true.get(rid) for rid in cluster if rid in record_to_true]

        if not true_ids:
            continue

        from collections import Counter

        counter = Counter(true_ids)
        majority_count = counter.most_common(1)[0][1]

        purity = majority_count / len(true_ids)
        purities.append(purity)

    avg_purity = np.mean(purities) if purities else 0.0

    logger.info(f"Cluster purity: {avg_purity:.3f}")

    return avg_purity


def generate_confusion_matrix_report(metrics: dict) -> str:
    """
    Generate a human-readable confusion matrix report.

    Args:
        metrics: Dictionary with evaluation metrics

    Returns:
        Formatted string report
    """
    report = "\n" + "=" * 50 + "\n"
    report += "Entity Resolution Evaluation Report\n"
    report += "=" * 50 + "\n\n"

    report += "Confusion Matrix:\n"
    report += f"  True Positives:  {metrics['true_positives']:6d}\n"
    report += f"  False Positives: {metrics['false_positives']:6d}\n"
    report += f"  False Negatives: {metrics['false_negatives']:6d}\n\n"

    report += "Metrics:\n"
    report += f"  Precision: {metrics['precision']:.4f}\n"
    report += f"  Recall:    {metrics['recall']:.4f}\n"
    report += f"  F1 Score:  {metrics['f1_score']:.4f}\n\n"

    report += "Pair Counts:\n"
    report += f"  Predicted pairs: {metrics.get('predicted_pairs', 'N/A')}\n"
    report += f"  True pairs:      {metrics.get('true_pairs', 'N/A')}\n"

    report += "=" * 50 + "\n"

    return report
