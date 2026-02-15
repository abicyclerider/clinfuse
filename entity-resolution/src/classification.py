"""
Classification of candidate pairs as matches or non-matches.

Supports multiple classification approaches:
- Threshold-based (simple sum of similarity scores)
- Probabilistic (logistic regression)
- LLM fallback (future extension for medical history comparison)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def classify_pairs(
    features: pd.DataFrame, config: dict, ground_truth: pd.DataFrame = None
) -> pd.Series:
    """
    Classify candidate pairs as matches (1) or non-matches (0).

    Args:
        features: DataFrame with similarity scores
        config: Configuration dictionary with classification method
        ground_truth: Ground truth for training (optional, required for logistic_regression)

    Returns:
        Boolean Series indicating matches
    """
    method = config.get("classification", {}).get("method", "threshold")

    if method == "threshold":
        return classify_threshold(features, config)

    elif method == "logistic_regression":
        if ground_truth is None:
            raise ValueError("Ground truth required for logistic_regression method")
        return classify_probabilistic(features, ground_truth, config)

    elif method == "tiered":
        return classify_tiered(features, config)

    else:
        raise ValueError(f"Unknown classification method: {method}")


def classify_threshold(features: pd.DataFrame, config: dict) -> pd.Series:
    """
    Threshold-based classification: sum similarity scores and apply cutoff.

    Args:
        features: DataFrame with similarity scores
        config: Configuration with threshold parameter

    Returns:
        Boolean Series indicating matches
    """
    threshold = config.get("classification", {}).get("threshold", 3.5)

    # Calculate total similarity score
    if "total_score" not in features.columns:
        from .comparison import add_composite_features

        features = add_composite_features(features)

    matches = features["total_score"] >= threshold

    logger.info(
        f"Threshold classification: {matches.sum()} matches at threshold {threshold}"
    )

    return matches


def classify_tiered(features: pd.DataFrame, config: dict) -> pd.Series:
    """
    Tiered classification with auto-match, auto-reject, and gray zone.

    Architecture:
    - auto-match:  score >= auto_match_threshold → match
    - gray zone:   score between auto_reject and auto_match → apply single_threshold
    - auto-reject: score < auto_reject_threshold → non-match

    Gray zone uses single_threshold as fallback (LLM matcher is Phase 5).

    Args:
        features: DataFrame with similarity scores
        config: Configuration with tiered thresholds

    Returns:
        Boolean Series indicating matches
    """
    cls_config = config.get("classification", {})
    auto_reject = cls_config.get("auto_reject_threshold", 4.0)
    auto_match = cls_config.get("auto_match_threshold", 6.0)
    single_threshold = cls_config.get("single_threshold", 5.60)

    # Ensure total_score exists
    if "total_score" not in features.columns:
        from .comparison import add_composite_features

        features = add_composite_features(features)

    scores = features["total_score"]

    # Auto-match: score >= auto_match_threshold
    matches = scores >= auto_match

    # Gray zone: between auto_reject and auto_match, apply single_threshold fallback
    gray_zone = (scores >= auto_reject) & (scores < auto_match)
    matches = matches | (gray_zone & (scores >= single_threshold))

    # Auto-reject: score < auto_reject_threshold → already False

    logger.info(f"Tiered classification: {matches.sum()} matches")
    logger.info(f"  Auto-match (>={auto_match}): {(scores >= auto_match).sum()}")
    logger.info(f"  Gray zone ({auto_reject}-{auto_match}): {gray_zone.sum()}")
    logger.info(
        f"  Gray zone matches (>={single_threshold}): {(gray_zone & (scores >= single_threshold)).sum()}"
    )
    logger.info(f"  Auto-reject (<{auto_reject}): {(scores < auto_reject).sum()}")

    return matches


def classify_probabilistic(
    features: pd.DataFrame, ground_truth: pd.DataFrame, config: dict
) -> pd.Series:
    """
    Probabilistic classification using logistic regression.

    Args:
        features: DataFrame with similarity scores
        ground_truth: Ground truth with true matches
        config: Configuration with train_test_split ratio

    Returns:
        Boolean Series indicating matches
    """
    # Label features with ground truth
    labeled_features = label_features_with_ground_truth(features, ground_truth)

    # Split into train/test
    split_ratio = config.get("classification", {}).get("train_test_split", 0.8)
    X = labeled_features.drop("is_match", axis=1)
    y = labeled_features["is_match"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=split_ratio, random_state=42, stratify=y
    )

    # Train logistic regression
    logger.info(f"Training logistic regression on {len(X_train)} samples...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on test set
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

    # Predict on all features
    predictions = model.predict(features)
    matches = pd.Series(predictions, index=features.index, dtype=bool)

    logger.info(f"Probabilistic classification: {matches.sum()} matches")

    return matches


def label_features_with_ground_truth(
    features: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """
    Label feature vectors with ground truth (match=1, non-match=0).

    Args:
        features: DataFrame with similarity scores (MultiIndex of record pairs)
        ground_truth: Ground truth DataFrame

    Returns:
        DataFrame with added 'is_match' column
    """
    # Create ground truth pair set

    # For this to work, we need record_id mapping - assuming it's in ground_truth
    # Ground truth should have: facility_id, patient_id, true_patient_id, record_id
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

    # Label features
    features = features.copy()
    features["is_match"] = False

    for idx in features.index:
        pair = tuple(sorted([idx[0], idx[1]]))
        if pair in true_pairs:
            features.loc[idx, "is_match"] = True

    match_count = features["is_match"].sum()
    total_count = len(features)
    logger.info(
        f"Labeled {match_count} matches and {total_count - match_count} non-matches"
    )

    return features


def apply_deterministic_rules(features: pd.DataFrame) -> pd.Series:
    """
    Apply deterministic matching rules for high-confidence cases.

    Rules:
    - Exact SSN + Exact birthdate = definite match
    - Exact SSN + high name similarity = likely match

    Args:
        features: DataFrame with similarity scores

    Returns:
        Boolean Series indicating deterministic matches
    """
    matches = pd.Series(False, index=features.index)

    # Rule 1: SSN + birthdate exact match
    if "ssn_match" in features.columns and "birthdate_match" in features.columns:
        rule1 = (features["ssn_match"] == 1) & (features["birthdate_match"] == 1)
        matches |= rule1
        logger.debug(f"Rule 1 (SSN + DOB): {rule1.sum()} matches")

    # Rule 2: SSN + high name similarity
    if (
        "ssn_match" in features.columns
        and "first_name_sim" in features.columns
        and "last_name_sim" in features.columns
    ):
        rule2 = (
            (features["ssn_match"] == 1)
            & (features["first_name_sim"] >= 0.9)
            & (features["last_name_sim"] >= 0.9)
        )
        matches |= rule2
        logger.debug(f"Rule 2 (SSN + names): {rule2.sum()} matches")

    logger.info(f"Deterministic rules: {matches.sum()} matches")

    return matches


def tune_threshold(
    features: pd.DataFrame,
    ground_truth: pd.DataFrame,
    threshold_range: tuple = (2.0, 5.0, 0.1),
) -> dict:
    """
    Find optimal threshold by evaluating precision/recall across range.

    Args:
        features: DataFrame with similarity scores
        ground_truth: Ground truth for evaluation
        threshold_range: (min, max, step) for threshold values

    Returns:
        Dictionary with optimal threshold and metrics
    """
    from .comparison import add_composite_features

    if "total_score" not in features.columns:
        features = add_composite_features(features)

    labeled_features = label_features_with_ground_truth(features, ground_truth)

    thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_range[2])
    results = []

    for threshold in thresholds:
        matches = labeled_features["total_score"] >= threshold
        tp = ((labeled_features["is_match"] == True) & (matches == True)).sum()
        fp = ((labeled_features["is_match"] == False) & (matches == True)).sum()
        fn = ((labeled_features["is_match"] == True) & (matches == False)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matches": matches.sum(),
            }
        )

    results_df = pd.DataFrame(results)
    best_idx = results_df["f1"].idxmax()
    best_result = results_df.iloc[best_idx].to_dict()

    logger.info(
        f"Optimal threshold: {best_result['threshold']:.2f} (F1={best_result['f1']:.3f})"
    )

    return best_result
