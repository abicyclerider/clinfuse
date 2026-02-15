"""
Field-by-field comparison for candidate record pairs.

Calculates similarity scores using appropriate methods for each field type.
"""

import logging

import pandas as pd
import recordlinkage as rl

logger = logging.getLogger(__name__)


def build_comparison_features(
    pairs: pd.MultiIndex, df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """
    Calculate similarity scores for all candidate pairs.

    Args:
        pairs: MultiIndex of candidate pairs
        df: Patient DataFrame with record_id as index
        config: Configuration dictionary with comparison thresholds

    Returns:
        DataFrame with similarity scores for each field
    """
    # Ensure record_id is the index
    if "record_id" in df.columns and df.index.name != "record_id":
        df = df.set_index("record_id")

    comp_config = config.get("comparison", {})

    # Create custom comparator
    compare = create_custom_comparator(comp_config)

    # Compute comparison features
    logger.info(f"Computing comparison features for {len(pairs)} candidate pairs...")
    features = compare.compute(pairs, df)

    logger.info(f"Generated {len(features.columns)} comparison features")

    return features


def create_custom_comparator(config: dict) -> rl.Compare:
    """
    Create a recordlinkage Compare object matching the validated 8-feature setup.

    Features (8 total, max score = 8.0):
      1. first_name_sim  — Jaro-Winkler (continuous 0-1)
      2. last_name_sim   — Jaro-Winkler (continuous 0-1)
      3. address_sim     — Jaro-Winkler (continuous 0-1)
      4. city_sim        — Jaro-Winkler (continuous 0-1)
      5. state_match     — Exact (0 or 1)
      6. zip_match       — Exact (0 or 1)
      7. ssn_match       — Exact (0 or 1) by default, or fuzzy via config
      8. birthdate_match — Exact (0 or 1) by default, or fuzzy via config

    Args:
        config: Configuration dictionary with optional ssn_fuzzy/birthdate_fuzzy flags

    Returns:
        Configured Compare object
    """
    compare = rl.Compare()

    # 1. First name: Jaro-Winkler similarity (no threshold — continuous score)
    compare.string(
        "first_name", "first_name", method="jarowinkler", label="first_name_sim"
    )

    # 2. Last name: Jaro-Winkler similarity
    compare.string(
        "last_name", "last_name", method="jarowinkler", label="last_name_sim"
    )

    # 3. Address: Jaro-Winkler similarity
    compare.string("address", "address", method="jarowinkler", label="address_sim")

    # 4. City: Jaro-Winkler similarity
    compare.string("city", "city", method="jarowinkler", label="city_sim")

    # 5. State: Exact match
    compare.exact("state", "state", label="state_match")

    # 6. ZIP: Exact match
    compare.exact("zip", "zip", label="zip_match")

    # 7. SSN: Exact match (default) or fuzzy Levenshtein
    if config.get("ssn_fuzzy", False):
        compare.string("ssn", "ssn", method="levenshtein", label="ssn_match")
    else:
        compare.exact("ssn", "ssn", label="ssn_match")

    # 8. Birthdate: Exact match (default) or fuzzy date comparison
    if config.get("birthdate_fuzzy", False):
        compare.date("birthdate", "birthdate", label="birthdate_match")
    else:
        compare.exact("birthdate", "birthdate", label="birthdate_match")

    return compare


def calculate_custom_similarity(
    df1: pd.DataFrame, df2: pd.DataFrame, field: str, method: str = "jarowinkler"
) -> pd.Series:
    """
    Calculate custom similarity scores for a specific field.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        field: Field name to compare
        method: Similarity method (jarowinkler, levenshtein, etc.)

    Returns:
        Series of similarity scores
    """
    if method == "jarowinkler":
        from jellyfish import jaro_winkler_similarity

        return df1[field].combine(
            df2[field], lambda a, b: jaro_winkler_similarity(str(a), str(b))
        )

    elif method == "levenshtein":
        from Levenshtein import distance as levenshtein_distance

        return df1[field].combine(
            df2[field],
            lambda a, b: (
                1 - levenshtein_distance(str(a), str(b)) / max(len(str(a)), len(str(b)))
            ),
        )

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def analyze_similarity_distribution(
    features: pd.DataFrame, field: str, true_matches: pd.Series = None
) -> dict:
    """
    Analyze the distribution of similarity scores for a field.

    Args:
        features: DataFrame with similarity scores
        field: Field name to analyze
        true_matches: Boolean series indicating true matches (optional)

    Returns:
        Dictionary with distribution statistics
    """
    if field not in features.columns:
        raise ValueError(f"Field {field} not found in features")

    stats = {
        "field": field,
        "mean": features[field].mean(),
        "median": features[field].median(),
        "std": features[field].std(),
        "min": features[field].min(),
        "max": features[field].max(),
        "q25": features[field].quantile(0.25),
        "q75": features[field].quantile(0.75),
    }

    if true_matches is not None:
        stats["mean_matches"] = features.loc[true_matches, field].mean()
        stats["mean_non_matches"] = features.loc[~true_matches, field].mean()

    return stats


def add_composite_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite features combining multiple similarity scores.

    Args:
        features: DataFrame with individual similarity scores

    Returns:
        DataFrame with added composite features
    """
    features = features.copy()

    # Total similarity score (sum of the 8 base features only)
    feature_cols = [
        "first_name_sim",
        "last_name_sim",
        "address_sim",
        "city_sim",
        "state_match",
        "zip_match",
        "ssn_match",
        "birthdate_match",
    ]
    cols_present = [c for c in feature_cols if c in features.columns]
    features["total_score"] = features[cols_present].sum(axis=1)

    # Name similarity (average of first, last)
    name_cols = [
        c for c in ["first_name_sim", "last_name_sim"] if c in features.columns
    ]
    if name_cols:
        features["name_score"] = features[name_cols].mean(axis=1)

    # Address similarity (average of address, city, state, zip)
    addr_cols = [
        c
        for c in ["address_sim", "city_sim", "state_match", "zip_match"]
        if c in features.columns
    ]
    if addr_cols:
        features["address_score"] = features[addr_cols].mean(axis=1)

    # High confidence indicators
    if "ssn_match" in features.columns and "birthdate_match" in features.columns:
        features["high_confidence"] = (features["ssn_match"] == 1) & (
            features["birthdate_match"] == 1
        )

    return features
