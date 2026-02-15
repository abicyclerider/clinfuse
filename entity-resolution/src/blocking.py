"""
Blocking (indexing) strategies for entity resolution.

Reduces the O(n²) comparison space by generating candidate pairs
that are likely to be matches.
"""

import logging

import pandas as pd
import recordlinkage as rl

logger = logging.getLogger(__name__)


def create_candidate_pairs(
    df: pd.DataFrame, strategy: str = "lastname_state", window: int = 5
) -> pd.MultiIndex:
    """
    Generate candidate record pairs using specified blocking strategy.

    Args:
        df: Patient DataFrame with record_id as index
        strategy: Blocking strategy to use
        window: Window size for sorted neighborhood (if applicable)

    Returns:
        MultiIndex of candidate pairs (record_id_1, record_id_2)
    """
    # Ensure record_id is the index
    if "record_id" in df.columns and df.index.name != "record_id":
        df = df.set_index("record_id")

    indexer = rl.Index()

    if strategy == "lastname_state":
        # Block on exact match of last name and state
        logger.info("Using lastname_state blocking strategy")
        indexer.block(left_on=["last_name", "state"])

    elif strategy == "sorted_neighborhood":
        # Sorted neighborhood on last name (handles typos)
        logger.info(f"Using sorted_neighborhood blocking strategy (window={window})")
        indexer.sortedneighbourhood(left_on="last_name", window=window)

    elif strategy == "zip_birthyear":
        # Block on ZIP code and birth year
        logger.info("Using zip_birthyear blocking strategy")
        # Extract birth year
        df["birth_year"] = df["birthdate"].dt.year
        indexer.block(left_on=["zip", "birth_year"])

    elif strategy == "multipass":
        # Multiple blocking passes, union of results
        logger.info("Using multipass blocking strategy")
        indexer.block(left_on=["last_name", "state"])
        indexer.block(left_on=["zip", "state"])
        indexer.sortedneighbourhood(left_on="last_name", window=window)

    elif strategy == "aggressive_multipass":
        # Validated 4-pass strategy achieving 100% blocking recall
        logger.info("Using aggressive_multipass blocking strategy")
        if "birth_year" not in df.columns:
            df = df.copy()
            df["birth_year"] = pd.to_datetime(df["birthdate"]).dt.year
        indexer.block(left_on=["last_name", "state"])  # Pass 1
        indexer.block(left_on=["zip", "birth_year"])  # Pass 2
        indexer.sortedneighbourhood(left_on="last_name", window=7)  # Pass 3
        indexer.sortedneighbourhood(left_on="first_name", window=5)  # Pass 4

    else:
        raise ValueError(f"Unknown blocking strategy: {strategy}")

    # Generate candidate pairs
    pairs = indexer.index(df)

    logger.info(f"Generated {len(pairs)} candidate pairs from {len(df)} records")
    logger.info(f"Reduction: {1 - len(pairs) / (len(df) * (len(df) - 1) / 2):.2%}")

    return pairs


def evaluate_blocking_recall(
    pairs: pd.MultiIndex, ground_truth: pd.DataFrame, record_id_mapping: pd.DataFrame
) -> dict:
    """
    Calculate blocking recall: percentage of true matches that pass blocking.

    Args:
        pairs: Candidate pairs from blocking
        ground_truth: Ground truth DataFrame with facility_id, patient_id, true_patient_id
        record_id_mapping: DataFrame mapping record_id to facility_id and patient_id

    Returns:
        Dictionary with recall metrics
    """
    # Create set of candidate pairs
    candidate_set = set(pairs.tolist())

    # Generate true match pairs from ground truth
    true_pairs = generate_true_pairs_from_ground_truth(ground_truth, record_id_mapping)

    # Count how many true pairs are in candidate set
    true_pairs_found = 0
    for pair in true_pairs:
        # Check both orderings since pairs are unordered
        if pair in candidate_set or (pair[1], pair[0]) in candidate_set:
            true_pairs_found += 1

    recall = true_pairs_found / len(true_pairs) if true_pairs else 0

    metrics = {
        "candidate_pairs": len(pairs),
        "true_pairs": len(true_pairs),
        "true_pairs_found": true_pairs_found,
        "blocking_recall": recall,
    }

    logger.info(
        f"Blocking recall: {recall:.2%} ({true_pairs_found}/{len(true_pairs)} true pairs found)"
    )

    return metrics


def generate_true_pairs_from_ground_truth(
    ground_truth: pd.DataFrame, record_id_mapping: pd.DataFrame = None
) -> set:
    """
    Generate set of true matching pairs from ground truth.

    If ground_truth already has a 'record_id' column (added by pipeline),
    uses it directly. Otherwise merges with record_id_mapping.

    Args:
        ground_truth: DataFrame with true_patient_id and optionally record_id
        record_id_mapping: DataFrame with record_id, facility_id, id (optional)

    Returns:
        Set of tuples (record_id_1, record_id_2) for true matches
    """
    # Determine the true ID column
    true_id_col = (
        "true_patient_id"
        if "true_patient_id" in ground_truth.columns
        else "original_patient_uuid"
    )

    # If record_id not already in ground truth, merge with mapping
    if "record_id" not in ground_truth.columns and record_id_mapping is not None:
        gt_with_records = ground_truth.merge(
            record_id_mapping[["record_id", "facility_id", "id"]],
            left_on=["facility_id", "original_patient_uuid"],
            right_on=["facility_id", "id"],
            how="left",
        )
    else:
        gt_with_records = ground_truth

    # Group by true patient ID to find all record_ids for same patient
    true_pairs = set()
    for true_id, group in gt_with_records.groupby(true_id_col):
        record_ids = group["record_id"].dropna().tolist()

        # Generate all pairs within this group
        for i in range(len(record_ids)):
            for j in range(i + 1, len(record_ids)):
                # Store pairs in sorted order for consistency
                pair = tuple(sorted([record_ids[i], record_ids[j]]))
                true_pairs.add(pair)

    return true_pairs


def block_on_field(df: pd.DataFrame, field: str) -> pd.MultiIndex:
    """
    Simple blocking on a single field (for experimentation).

    Args:
        df: Patient DataFrame
        field: Field name to block on

    Returns:
        MultiIndex of candidate pairs
    """
    indexer = rl.Index()
    indexer.block(left_on=field)
    pairs = indexer.index(df)

    logger.info(f"Blocking on {field}: {len(pairs)} candidate pairs")

    return pairs


def analyze_blocking_statistics(df: pd.DataFrame, field: str) -> dict:
    """
    Analyze cardinality and distribution of a blocking field.

    Args:
        df: Patient DataFrame
        field: Field name to analyze

    Returns:
        Dictionary with statistics
    """
    if field not in df.columns:
        raise ValueError(f"Field {field} not found in DataFrame")

    stats = {
        "field": field,
        "total_records": len(df),
        "unique_values": df[field].nunique(),
        "cardinality_ratio": df[field].nunique() / len(df),
        "null_count": df[field].isna().sum(),
        "max_block_size": df[field].value_counts().max(),
        "avg_block_size": len(df) / df[field].nunique()
        if df[field].nunique() > 0
        else 0,
    }

    return stats
