"""
Shared evaluation metrics for entity resolution.

Provides core confusion matrix and metric calculations.
"""

from typing import Tuple


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
