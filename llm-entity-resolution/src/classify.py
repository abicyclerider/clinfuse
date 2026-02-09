"""
Hybrid classification: classical demographics + LLM medical records.

Sequential gating architecture:
  - Auto-reject (score < 4.0): kept as non-match
  - Auto-match (score >= 6.0): kept as match
  - Gray zone (4.0-6.0): LLM compares medical records only

Usage:
    cd llm-entity-resolution
    python -m src.classify --config config/llm_config.yaml
"""

import sys
from pathlib import Path
import logging
import argparse
import json

import os
import re

import dspy
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.data_loader import load_facility_patients, get_run_directory
from shared.ground_truth import (
    load_ground_truth,
    add_record_ids_to_ground_truth,
    generate_true_pairs_from_ground_truth,
)
from shared.evaluation import evaluate_matches, calculate_metrics
from shared.medical_records import load_medical_records

from .dspy_modules import MedicalRecordMatcher
from .summarize import summarize_patient_records
from .langfuse_setup import init_langfuse
from .utils import load_config

logger = logging.getLogger(__name__)


def load_classical_results(config: dict) -> pd.DataFrame:
    """
    Load the classical pipeline's predicted_matches.csv with scores.

    This CSV contains all candidate pairs that passed blocking, with their
    similarity features and total_score.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with columns: record_id_1, record_id_2, total_score, ...
    """
    gz_config = config.get('gray_zone', {})
    features_csv = gz_config.get('features_csv', '../entity-resolution/output/predicted_matches.csv')

    # Resolve path
    features_path = Path(config['base_dir']) / features_csv.lstrip('../')
    if not features_path.exists():
        features_path = Path(__file__).resolve().parent.parent / features_csv

    logger.info(f"Loading classical results from {features_path}")
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} candidate pairs")

    return df


def classify_with_llm(gray_zone_pairs: pd.DataFrame, config: dict,
                       medical_records: dict[str, pd.DataFrame],
                       record_map: dict,
                       matcher: MedicalRecordMatcher) -> pd.DataFrame:
    """
    Run LLM classification on gray zone pairs using medical records only.

    Args:
        gray_zone_pairs: DataFrame of gray zone pairs with record_id_1, record_id_2
        config: Configuration dict
        medical_records: Loaded medical records from all facilities
        record_map: Dict mapping record_id → (patient_uuid, facility_id)
        matcher: DSPy MedicalRecordMatcher module

    Returns:
        DataFrame with added columns: llm_is_match, llm_confidence, llm_reasoning
    """
    gz_config = config.get('gray_zone', {})
    confidence_threshold = gz_config.get('confidence_threshold', 0.7)

    results = []
    total = len(gray_zone_pairs)

    for i, (_, row) in enumerate(gray_zone_pairs.iterrows()):
        rid1, rid2 = row['record_id_1'], row['record_id_2']

        if rid1 not in record_map or rid2 not in record_map:
            logger.warning(f"Missing record map for {rid1} or {rid2}")
            results.append({
                'llm_is_match': False,
                'llm_confidence': 0.0,
                'llm_reasoning': 'record_id not found in patient data',
            })
            continue

        uuid1, fac1 = record_map[rid1]
        uuid2, fac2 = record_map[rid2]

        # Generate medical history summaries
        summary_a = summarize_patient_records(uuid1, fac1, medical_records)
        summary_b = summarize_patient_records(uuid2, fac2, medical_records)

        try:
            pred = matcher(
                medical_history_a=summary_a,
                medical_history_b=summary_b,
            )

            is_match = bool(pred.is_match)
            confidence = float(pred.confidence)
            reasoning = str(getattr(pred, 'reasoning', ''))

            # Apply confidence threshold
            if confidence < confidence_threshold:
                is_match = False  # Fall back to non-match if uncertain

        except Exception as e:
            logger.error(f"LLM error on pair ({rid1}, {rid2}): {e}")
            is_match = False
            confidence = 0.0
            reasoning = f"LLM error: {e}"

        results.append({
            'llm_is_match': is_match,
            'llm_confidence': confidence,
            'llm_reasoning': reasoning,
        })

        if (i + 1) % 5 == 0 or (i + 1) == total:
            logger.info(f"  Processed {i + 1}/{total} gray zone pairs")

    results_df = pd.DataFrame(results)

    # Concatenate with original gray zone pairs
    gray_zone_with_llm = pd.concat(
        [gray_zone_pairs.reset_index(drop=True), results_df],
        axis=1,
    )

    match_count = results_df['llm_is_match'].sum()
    logger.info(f"LLM classified {match_count}/{total} gray zone pairs as matches")

    return gray_zone_with_llm


def run_hybrid_classification(config: dict):
    """
    Run the full hybrid classification pipeline.

    1. Load classical pipeline output
    2. Split into auto-reject, auto-match, gray zone
    3. Run LLM on gray zone pairs
    4. Merge all decisions
    5. Evaluate against ground truth

    Args:
        config: Configuration dictionary
    """
    # Initialize Langfuse tracing
    init_langfuse(config)

    # Load .env for API keys
    load_dotenv()

    # Configure DSPy
    model_config = config['model']

    # Resolve API key: support api_key_env (env var name) or direct api_key
    if 'api_key_env' in model_config:
        api_key = os.environ.get(model_config['api_key_env'], '')
        if not api_key:
            raise ValueError(f"${model_config['api_key_env']} not set in environment")
    else:
        api_key = model_config['api_key']

    # Resolve api_base: substitute ${VAR} references from environment
    api_base = model_config['api_base']
    api_base = re.sub(
        r'\$\{(\w+)\}',
        lambda m: os.environ.get(m.group(1), m.group(0)),
        api_base,
    )

    lm = dspy.LM(
        model=f"openai/{model_config['name']}",
        api_base=api_base,
        api_key=api_key,
        temperature=model_config.get('temperature', 0.1),
        max_tokens=model_config.get('max_tokens', 256),
    )
    dspy.configure(lm=lm)

    # Load and prepare matcher
    matcher = MedicalRecordMatcher()
    dspy_config = config.get('dspy', {})
    optimized_path = Path(__file__).resolve().parent.parent / dspy_config.get(
        'optimized_program', 'data/dspy/optimized_program.json')

    if optimized_path.exists():
        matcher.load(str(optimized_path))
        logger.info(f"Loaded optimized program from {optimized_path}")
    else:
        logger.info("No optimized program found — using baseline module")

    # Load classical pipeline results
    classical_df = load_classical_results(config)

    # Split by thresholds
    gz_config = config.get('gray_zone', {})
    lower = gz_config.get('lower_threshold', 4.0)
    upper = gz_config.get('upper_threshold', 6.0)

    auto_reject = classical_df[classical_df['total_score'] < lower].copy()
    auto_match = classical_df[classical_df['total_score'] >= upper].copy()
    gray_zone = classical_df[
        (classical_df['total_score'] >= lower) &
        (classical_df['total_score'] < upper)
    ].copy()

    logger.info(f"Auto-reject (<{lower}): {len(auto_reject)} pairs")
    logger.info(f"Auto-match (>={upper}): {len(auto_match)} pairs")
    logger.info(f"Gray zone ({lower}-{upper}): {len(gray_zone)} pairs")

    # Assign classical decisions
    auto_reject['is_match'] = False
    auto_reject['decision_source'] = 'auto_reject'
    auto_match['is_match'] = True
    auto_match['decision_source'] = 'auto_match'

    # Run LLM on gray zone
    if len(gray_zone) > 0:
        run_dir = get_run_directory(config['base_dir'], config['run_id'])

        # Load patient data for record_id mapping
        patients_df = load_facility_patients(str(run_dir))
        patients_df['record_id'] = patients_df['facility_id'] + '_' + patients_df['Id'].astype(str)

        record_map = {}
        for _, row in patients_df.iterrows():
            record_map[row['record_id']] = (row['Id'], row['facility_id'])

        # Load medical records
        logger.info("Loading medical records...")
        medical_records = load_medical_records(str(run_dir))

        # Classify gray zone pairs
        gray_zone = classify_with_llm(
            gray_zone, config, medical_records, record_map, matcher)

        gray_zone['is_match'] = gray_zone['llm_is_match']
        gray_zone['decision_source'] = 'llm'
    else:
        gray_zone['is_match'] = False
        gray_zone['decision_source'] = 'gray_zone_empty'

    # Merge all decisions
    all_pairs = pd.concat([auto_reject, auto_match, gray_zone], ignore_index=True)
    logger.info(f"Total pairs: {len(all_pairs)}, matches: {all_pairs['is_match'].sum()}")

    # Evaluate against ground truth
    run_dir = get_run_directory(config['base_dir'], config['run_id'])
    patients_df = load_facility_patients(str(run_dir))
    patients_df['record_id'] = patients_df['facility_id'] + '_' + patients_df['Id'].astype(str)

    ground_truth_df = load_ground_truth(str(run_dir))
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    true_pairs = generate_true_pairs_from_ground_truth(ground_truth_df)

    # Build predicted pairs set
    matched = all_pairs[all_pairs['is_match']]
    predicted_pairs = set()
    for _, row in matched.iterrows():
        pair = tuple(sorted([row['record_id_1'], row['record_id_2']]))
        predicted_pairs.add(pair)

    metrics = evaluate_matches(predicted_pairs, true_pairs)

    # Print results
    print("\n" + "=" * 60)
    print("Hybrid Classification Results (Classical + LLM)")
    print("=" * 60)
    print(f"\n  Total pairs evaluated:  {len(all_pairs)}")
    print(f"  Auto-reject:            {len(auto_reject)}")
    print(f"  Auto-match:             {len(auto_match)}")
    print(f"  Gray zone (LLM):        {len(gray_zone)}")
    print(f"  Gray zone → match:      {gray_zone['is_match'].sum() if 'is_match' in gray_zone else 0}")
    print(f"\n  Predicted matches:      {metrics['predicted_pairs']}")
    print(f"  True pairs:             {metrics['true_pairs']}")
    print(f"  True positives:         {metrics['true_positives']}")
    print(f"  False positives:        {metrics['false_positives']}")
    print(f"  False negatives:        {metrics['false_negatives']}")
    print(f"\n  Precision:              {metrics['precision']:.4f}")
    print(f"  Recall:                 {metrics['recall']:.4f}")
    print(f"  F1 Score:               {metrics['f1_score']:.4f}")
    print("=" * 60 + "\n")

    # Save results
    output_dir = Path(__file__).resolve().parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs.to_csv(output_dir / 'hybrid_predictions.csv', index=False)
    logger.info(f"Saved hybrid predictions to {output_dir / 'hybrid_predictions.csv'}")

    with open(output_dir / 'hybrid_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {output_dir / 'hybrid_metrics.json'}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run hybrid entity resolution classification")
    parser.add_argument('--config', default='config/llm_config.yaml',
                        help='Path to configuration YAML')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    config = load_config(args.config)
    run_hybrid_classification(config)


if __name__ == '__main__':
    main()
