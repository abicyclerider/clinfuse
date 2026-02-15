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
