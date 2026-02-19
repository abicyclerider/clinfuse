"""
Splink v4 probabilistic record linkage for entity resolution.

Replaces recordlinkage-based blocking + comparison + scoring with
Fellegi-Sunter model: EM-estimated weights, term frequency adjustments,
and calibrated match probabilities.
"""

import logging
import math

import pandas as pd
import splink.comparison_library as cl
from splink import ColumnExpression, DuckDBAPI, Linker, SettingsCreator, block_on

logger = logging.getLogger(__name__)


def build_settings(config: dict) -> SettingsCreator:
    """Build Splink SettingsCreator with comparisons and blocking rules.

    Comparisons (7 fields):
      - first_name, last_name: JaroWinkler with TF adjustment
      - address: JaroWinkler
      - city: JaroWinkler with TF adjustment
      - zip, ssn: ExactMatch
      - birthdate: DateOfBirthComparison with date proximity tiers

    Note: `state` is excluded because all records are in Massachusetts,
    giving it zero discriminating power. Including it would leave Splink
    unable to estimate u/m probabilities, degrading calibration.
    """
    comparisons = [
        cl.JaroWinklerAtThresholds("first_name", [0.92, 0.80]).configure(
            term_frequency_adjustments=True
        ),
        cl.JaroWinklerAtThresholds("last_name", [0.92, 0.80]).configure(
            term_frequency_adjustments=True
        ),
        cl.JaroWinklerAtThresholds("address", [0.9, 0.7]),
        cl.JaroWinklerAtThresholds("city", [0.9, 0.7]).configure(
            term_frequency_adjustments=True
        ),
        cl.ExactMatch("zip"),
        cl.ExactMatch("ssn"),
        cl.DateOfBirthComparison("birthdate", input_is_string=True),
    ]

    # Substring expressions for fuzzy blocking (catches typos in names)
    last_name_3 = ColumnExpression("substr(last_name, 1, 3)")
    first_name_3 = ColumnExpression("substr(first_name, 1, 3)")

    blocking_rules = [
        block_on("last_name", "city"),
        block_on("zip", "birth_year"),
        block_on("first_name", "birth_year"),
        block_on("ssn"),
        block_on("birthdate"),
        block_on("first_name", "zip"),
        block_on("address"),
        # Fuzzy blocking: catch pairs where name typos break exact-match rules
        block_on(last_name_3, "birth_year", "zip"),
        block_on(first_name_3, "last_name"),
        # Broad single-field rules to catch heavily-errored records
        block_on("last_name"),
        block_on("birth_year"),
        block_on("first_name"),
        block_on(first_name_3, "birth_year"),
        block_on(last_name_3, "birth_year"),
        block_on("zip"),
        block_on("city"),
    ]

    splink_cfg = config.get("splink", {})
    predict_threshold = splink_cfg.get("predict_threshold", 0.01)

    settings = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="record_id",
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )

    return settings, predict_threshold


def create_linker(df: pd.DataFrame, config: dict) -> tuple:
    """Create a Splink Linker with DuckDB backend.

    Args:
        df: Patient DataFrame with record_id column (not as index).
        config: Matching config dict.

    Returns:
        (linker, predict_threshold) tuple.
    """
    settings, predict_threshold = build_settings(config)
    linker = Linker(df, settings, db_api=DuckDBAPI())
    return linker, predict_threshold


def train_model(linker: Linker) -> None:
    """Train Splink model with unsupervised EM estimation.

    Three-step process:
      1. Estimate lambda (P(two random records match)) using deterministic rules
      2. Estimate u probabilities using random sampling
      3. Estimate m probabilities with rotating EM blocking
    """
    # Step 1: Estimate probability two random records match
    deterministic_rules = [
        block_on("ssn"),
        block_on("first_name", "last_name", "birthdate"),
    ]
    logger.info("Estimating probability two random records match...")
    linker.training.estimate_probability_two_random_records_match(
        deterministic_rules, recall=0.7
    )

    # Step 2: Estimate u probabilities from random sample
    logger.info("Estimating u probabilities from random sampling...")
    linker.training.estimate_u_using_random_sampling(max_pairs=5e5)

    # Step 3: EM sessions with rotating blocking to estimate m probabilities.
    # Each session estimates m for all comparisons EXCEPT those in the blocking rule.
    # We need enough sessions so every comparison gets m estimated at least once.
    #
    # Session 1 (block on birthdate): estimates first_name, last_name, address, city, zip, ssn
    # Session 2 (block on last_name, city): estimates first_name, address, zip, ssn, birthdate
    # Session 3 (block on ssn): estimates first_name, last_name, address, city, zip, birthdate
    logger.info("EM session 1: blocking on birthdate...")
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("birthdate"),
        fix_u_probabilities=True,
    )

    logger.info("EM session 2: blocking on last_name + city...")
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("last_name", "city"),
        fix_u_probabilities=True,
    )

    logger.info("EM session 3: blocking on ssn...")
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("ssn"),
        fix_u_probabilities=True,
    )

    logger.info("Splink model training complete.")


def predict_matches(linker: Linker, config: dict) -> pd.DataFrame:
    """Run Splink prediction and return results as pandas DataFrame.

    Returns DataFrame with columns:
      record_id_l, record_id_r, match_probability, match_weight,
      gamma_*, bf_* (intermediate columns)
    """
    splink_cfg = config.get("splink", {})
    predict_threshold = splink_cfg.get("predict_threshold", 0.01)

    logger.info(f"Predicting matches (threshold={predict_threshold})...")
    df_predictions = linker.inference.predict(
        threshold_match_probability=predict_threshold
    )
    results = df_predictions.as_pandas_dataframe()
    logger.info(f"Splink produced {len(results)} candidate pairs above threshold")

    return results


def classify_predictions(predictions_df: pd.DataFrame, config: dict) -> tuple:
    """Split Splink predictions into auto_match / gray_zone / auto_reject.

    Args:
        predictions_df: DataFrame from predict_matches() with match_probability.
        config: Config dict with splink.auto_match_probability and
                splink.auto_reject_probability.

    Returns:
        (auto_matches_df, gray_zone_df, all_predictions_df) tuple.
        Each has columns: record_id_1, record_id_2, match_probability, match_weight, ...
    """
    splink_cfg = config.get("splink", {})
    auto_match_prob = splink_cfg.get("auto_match_probability", 0.95)
    auto_reject_prob = splink_cfg.get("auto_reject_probability", 0.05)

    # Normalize column names: Splink uses record_id_l/r, we want record_id_1/2
    df = predictions_df.copy()
    df = df.rename(
        columns={
            "record_id_l": "record_id_1",
            "record_id_r": "record_id_2",
        }
    )

    # For backward compat, map match_probability to total_score
    df["total_score"] = df["match_probability"]

    auto_match_mask = df["match_probability"] >= auto_match_prob
    auto_reject_mask = df["match_probability"] < auto_reject_prob
    gray_zone_mask = ~auto_match_mask & ~auto_reject_mask

    auto_matches = df[auto_match_mask].copy()
    gray_zone = df[gray_zone_mask].copy()
    auto_reject_count = int(auto_reject_mask.sum())

    logger.info(
        f"Classification: auto_match={len(auto_matches)} "
        f"(P>={auto_match_prob}), gray_zone={len(gray_zone)}, "
        f"auto_reject={auto_reject_count} (P<{auto_reject_prob})"
    )

    return auto_matches, gray_zone, df


def evaluate_splink_only(
    predictions_df: pd.DataFrame, true_pairs: set, config: dict
) -> dict:
    """Evaluate Splink-only performance: auto-match F1 and best single-threshold F1.

    Args:
        predictions_df: All predictions with record_id_1, record_id_2, match_probability.
        true_pairs: Set of (record_id_1, record_id_2) tuples (sorted) from ground truth.
        config: Config dict.

    Returns:
        Dict with splink_only_auto_f1, splink_only_best_f1, splink_only_best_threshold.
    """
    splink_cfg = config.get("splink", {})
    auto_match_prob = splink_cfg.get("auto_match_probability", 0.95)

    # Build predicted pair sets at auto-match threshold
    auto_pairs = set()
    for _, row in predictions_df.iterrows():
        if row["match_probability"] >= auto_match_prob:
            pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
            auto_pairs.add(pair)

    tp_auto = len(auto_pairs & true_pairs)
    fp_auto = len(auto_pairs - true_pairs)
    fn_auto = len(true_pairs - auto_pairs)
    p_auto = tp_auto / (tp_auto + fp_auto) if (tp_auto + fp_auto) > 0 else 0
    r_auto = tp_auto / (tp_auto + fn_auto) if (tp_auto + fn_auto) > 0 else 0
    f1_auto = 2 * p_auto * r_auto / (p_auto + r_auto) if (p_auto + r_auto) > 0 else 0

    # Threshold sweep for best single-threshold F1
    thresholds = [i / 100 for i in range(1, 100)]
    best_f1 = 0
    best_thresh = 0.5

    # Pre-build pair list for efficiency
    pred_pairs = []
    for _, row in predictions_df.iterrows():
        pair = tuple(sorted([row["record_id_1"], row["record_id_2"]]))
        pred_pairs.append((pair, row["match_probability"]))

    for thresh in thresholds:
        predicted = {pair for pair, prob in pred_pairs if prob >= thresh}
        tp = len(predicted & true_pairs)
        fp = len(predicted - true_pairs)
        fn = len(true_pairs - predicted)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    metrics = {
        "splink_only_auto_f1": round(f1_auto, 4),
        "splink_only_auto_precision": round(p_auto, 4),
        "splink_only_auto_recall": round(r_auto, 4),
        "splink_only_best_f1": round(best_f1, 4),
        "splink_only_best_threshold": round(best_thresh, 2),
    }

    logger.info(
        f"Splink-only auto-match F1={f1_auto:.4f} (P={p_auto:.4f}, R={r_auto:.4f})"
    )
    logger.info(f"Splink-only best threshold={best_thresh:.2f}, F1={best_f1:.4f}")

    return metrics


def splink_logit(match_probability: float) -> float:
    """Convert Splink match probability to log-odds (logit)."""
    mp = max(1e-4, min(1 - 1e-4, match_probability))
    return math.log(mp / (1.0 - mp))
