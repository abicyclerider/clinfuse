"""
Shared evaluation metrics for entity resolution.

Compares predicted matches against ground truth to calculate precision, recall, F1.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def evaluate_matches(predicted_pairs: set, true_pairs: set) -> dict:
    """
    Evaluate predicted match pairs against ground truth pairs.

    Args:
        predicted_pairs: Set of predicted matching pairs (record_id_1, record_id_2)
        true_pairs: Set of true matching pairs (record_id_1, record_id_2)

    Returns:
        Dictionary with evaluation metrics
    """
    tp, fp, fn = calculate_confusion_matrix(predicted_pairs, true_pairs)
    metrics = calculate_metrics(tp, fp, fn)

    metrics["predicted_pairs"] = len(predicted_pairs)
    metrics["true_pairs"] = len(true_pairs)

    logger.info(
        f"Evaluation: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}"
    )

    return metrics


def calculate_confusion_matrix(
    predicted_pairs: set, true_pairs: set
) -> Tuple[int, int, int]:
    """
    Calculate confusion matrix components.

    Args:
        predicted_pairs: Set of predicted matching pairs
        true_pairs: Set of true matching pairs

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    predicted_normalized = {tuple(sorted(p)) for p in predicted_pairs}
    true_normalized = {tuple(sorted(p)) for p in true_pairs}

    tp = len(predicted_normalized & true_normalized)
    fp = len(predicted_normalized - true_normalized)
    fn = len(true_normalized - predicted_normalized)

    return tp, fp, fn


def calculate_metrics(tp: int, fp: int, fn: int) -> dict:
    """
    Calculate precision, recall, and F1 score from confusion matrix.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        Dictionary with precision, recall, f1_score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
