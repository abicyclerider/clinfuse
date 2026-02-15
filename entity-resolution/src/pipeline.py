"""
End-to-end entity resolution pipeline orchestration.

Coordinates the workflow: load → block → compare → classify → merge → evaluate.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from .blocking import create_candidate_pairs, evaluate_blocking_recall
from .classification import classify_pairs
from .comparison import add_composite_features, build_comparison_features
from .data_loader import load_data_for_matching
from .evaluation import (
    error_analysis,
    evaluate_golden_records,
    evaluate_matches,
    generate_confusion_matrix_report,
)
from .golden_record import create_golden_records

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    return config


def run_entity_resolution_pipeline(
    config: dict, output_dir: str = None, verbose: bool = False
) -> dict:
    """
    Run the complete entity resolution pipeline.

    Args:
        config: Configuration dictionary
        output_dir: Directory for output files (optional)
        verbose: Enable verbose logging

    Returns:
        Dictionary with results and metrics
    """
    # Setup logging
    setup_logging(verbose)

    logger.info("=" * 60)
    logger.info("Starting Entity Resolution Pipeline")
    logger.info("=" * 60)

    # Set output directory
    if output_dir is None:
        output_dir = config.get("output", {}).get("output_dir", "output")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Step 1: Load data
    logger.info("\n[Step 1/6] Loading data...")
    patients_df, ground_truth_df = load_data_for_matching(config)
    results["num_records"] = len(patients_df)
    results["num_facilities"] = patients_df["facility_id"].nunique()

    # Prepare ground truth with record_ids
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    # Step 2: Blocking
    logger.info("\n[Step 2/6] Generating candidate pairs (blocking)...")
    blocking_config = config.get("blocking", {})
    strategy = blocking_config.get("strategy", "lastname_state")
    window = blocking_config.get("sorted_neighborhood_window", 5)

    # Set index for record linkage
    patients_indexed = patients_df.set_index("record_id")

    candidate_pairs = create_candidate_pairs(
        patients_indexed, strategy=strategy, window=window
    )
    results["candidate_pairs"] = len(candidate_pairs)

    # Evaluate blocking recall
    record_id_mapping = patients_df[["record_id", "facility_id", "id"]]
    blocking_metrics = evaluate_blocking_recall(
        candidate_pairs, ground_truth_df, record_id_mapping
    )
    results["blocking_recall"] = blocking_metrics["blocking_recall"]

    # Step 3: Comparison
    logger.info("\n[Step 3/6] Computing similarity features...")
    features = build_comparison_features(candidate_pairs, patients_indexed, config)
    features = add_composite_features(features)
    results["num_features"] = len(features.columns)

    # Step 4: Classification
    logger.info("\n[Step 4/6] Classifying candidate pairs...")
    matches = classify_pairs(features, config, ground_truth_df)
    results["predicted_matches"] = matches.sum()

    # Step 5: Golden record creation
    logger.info("\n[Step 5/6] Creating golden records...")
    golden_records_df = create_golden_records(matches, patients_df, config)
    results["num_golden_records"] = len(golden_records_df)

    # Step 6: Evaluation
    logger.info("\n[Step 6/6] Evaluating results...")
    eval_metrics = evaluate_matches(matches, ground_truth_df, patients_df)
    results.update(eval_metrics)

    # Golden record evaluation
    golden_metrics = evaluate_golden_records(golden_records_df, ground_truth_df)
    results.update(golden_metrics)

    # Error analysis
    if verbose:
        error_results = error_analysis(matches, ground_truth_df, features, patients_df)
        results["error_analysis"] = {
            "fp_count": error_results["fp_count"],
            "fn_count": error_results["fn_count"],
        }

    # Save results
    logger.info("\n[Saving Results]")
    save_results(golden_records_df, matches, features, results, output_path, config)

    # Print summary
    print(generate_confusion_matrix_report(results))

    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)

    return results


def add_record_ids_to_ground_truth(
    ground_truth_df: pd.DataFrame, patients_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add record_id column to ground truth by merging with patient data.

    Args:
        ground_truth_df: Ground truth DataFrame
        patients_df: Patient DataFrame with record_id

    Returns:
        Ground truth DataFrame with record_id column
    """
    # Create mapping from facility_id + id to record_id
    mapping = patients_df[["facility_id", "id", "record_id"]].copy()

    # Ground truth has original_patient_uuid (the patient's UUID within each facility)
    # Patient data has id (same UUID). Merge on facility_id + these UUIDs.
    gt_with_records = ground_truth_df.merge(
        mapping,
        left_on=["facility_id", "original_patient_uuid"],
        right_on=["facility_id", "id"],
        how="left",
    )

    # Drop the duplicate 'id' column if it exists
    if "id" in gt_with_records.columns:
        gt_with_records = gt_with_records.drop("id", axis=1)

    # Rename to true_patient_id for downstream consistency
    gt_with_records = gt_with_records.rename(
        columns={"original_patient_uuid": "true_patient_id"}
    )

    return gt_with_records


def save_results(
    golden_records_df: pd.DataFrame,
    matches: pd.Series,
    features: pd.DataFrame,
    metrics: dict,
    output_path: Path,
    config: dict,
):
    """
    Save all results to output directory.

    Args:
        golden_records_df: Golden records DataFrame
        matches: Boolean Series of matches
        features: Features DataFrame
        metrics: Evaluation metrics dictionary
        output_path: Output directory Path
        config: Configuration dictionary
    """
    output_config = config.get("output", {})

    # Save golden records
    golden_file = output_path / output_config.get(
        "golden_records_csv", "golden_records.csv"
    )
    golden_records_df.to_csv(golden_file, index=False)
    logger.info(f"Saved golden records to {golden_file}")

    # Save predicted matches
    matches_file = output_path / output_config.get(
        "matches_csv", "predicted_matches.csv"
    )
    matches_df = pd.DataFrame(
        {
            "record_id_1": [pair[0] for pair in matches[matches].index],
            "record_id_2": [pair[1] for pair in matches[matches].index],
        }
    )

    # Add similarity scores if available
    if len(matches_df) > 0:
        for col in features.columns:
            if col not in ["is_match"]:
                match_indices = matches[matches].index
                matches_df[col] = [
                    features.loc[idx, col]
                    if idx in features.index
                    else features.loc[(idx[1], idx[0]), col]
                    if (idx[1], idx[0]) in features.index
                    else None
                    for idx in match_indices
                ]

    matches_df.to_csv(matches_file, index=False)
    logger.info(f"Saved predicted matches to {matches_file}")

    # Save evaluation metrics
    metrics_file = output_path / output_config.get(
        "metrics_json", "evaluation_metrics.json"
    )
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved evaluation metrics to {metrics_file}")


def setup_logging(verbose: bool = False):
    """
    Setup logging configuration.

    Args:
        verbose: Enable debug level logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_config(config: dict) -> bool:
    """
    Validate configuration has all required fields.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_fields = ["run_id", "base_dir"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")

    return True


def get_pipeline_summary(results: dict) -> str:
    """
    Generate a summary of pipeline results.

    Args:
        results: Results dictionary

    Returns:
        Formatted summary string
    """
    summary = "\n" + "=" * 60 + "\n"
    summary += "Pipeline Summary\n"
    summary += "=" * 60 + "\n\n"

    summary += "Data:\n"
    summary += f"  Total records:     {results.get('num_records', 'N/A')}\n"
    summary += f"  Facilities:        {results.get('num_facilities', 'N/A')}\n\n"

    summary += "Blocking:\n"
    summary += f"  Candidate pairs:   {results.get('candidate_pairs', 'N/A')}\n"
    summary += f"  Blocking recall:   {results.get('blocking_recall', 0):.2%}\n\n"

    summary += "Classification:\n"
    summary += f"  Predicted matches: {results.get('predicted_matches', 'N/A')}\n"
    summary += f"  True pairs:        {results.get('true_pairs', 'N/A')}\n\n"

    summary += "Golden Records:\n"
    summary += f"  Created:           {results.get('num_golden_records', 'N/A')}\n"
    summary += f"  Expected:          {results.get('num_true_patients', 'N/A')}\n\n"

    summary += "Performance:\n"
    summary += f"  Precision:         {results.get('precision', 0):.4f}\n"
    summary += f"  Recall:            {results.get('recall', 0):.4f}\n"
    summary += f"  F1 Score:          {results.get('f1_score', 0):.4f}\n"

    summary += "=" * 60 + "\n"

    return summary
