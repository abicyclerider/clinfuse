"""
DSPy optimization for medical record entity resolution.

Loads gray zone pairs from the classical pipeline, generates medical history
summaries, and runs MIPROv2 to optimize the MedicalRecordMatcher module.

Usage:
    cd llm-entity-resolution
    python -m src.optimize [--config config/llm_config.yaml]
"""

import sys
from pathlib import Path
import logging
import argparse

import os
import re

from dotenv import load_dotenv
import dspy
import pandas as pd

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.data_loader import load_facility_patients, get_run_directory
from shared.ground_truth import load_ground_truth, add_record_ids_to_ground_truth
from shared.medical_records import load_medical_records

from .dspy_modules import MedicalRecordMatcher
from .summarize import summarize_patient_records
from .utils import load_config, extract_answer

logger = logging.getLogger(__name__)


def accuracy_metric(example, pred, trace=None):
    """DSPy metric: does the prediction match the ground truth label?"""
    return pred.is_match == example.is_match


def build_training_data(config: dict) -> list[dspy.Example]:
    """
    Build DSPy training examples from gray zone pairs.

    1. Load classical pipeline's predicted_matches.csv → filter to gray zone
    2. Join with ground truth to get labels
    3. Generate medical history summaries for each pair
    4. Return as dspy.Example objects

    Args:
        config: Configuration dictionary

    Returns:
        List of dspy.Example with inputs (medical_history_a, medical_history_b)
        and label (is_match)
    """
    run_dir = get_run_directory(config['base_dir'], config['run_id'])

    # Load patient data to map record_ids → patient UUIDs
    patients_df = load_facility_patients(str(run_dir))
    patients_df['record_id'] = patients_df['facility_id'] + '_' + patients_df['Id'].astype(str)

    # Load ground truth
    ground_truth_df = load_ground_truth(str(run_dir))
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    # Build set of true matching pairs
    true_pairs = set()
    true_id_col = 'true_patient_id'
    for true_id, group in ground_truth_df.groupby(true_id_col):
        rids = group['record_id'].dropna().tolist()
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                true_pairs.add(tuple(sorted([rids[i], rids[j]])))

    # Load gray zone pairs from classical pipeline output
    gz_config = config.get('gray_zone', {})
    features_csv = gz_config.get('features_csv', '../entity-resolution/output/predicted_matches.csv')
    features_path = Path(config['base_dir']) / features_csv.lstrip('../')

    if not features_path.exists():
        # Try relative to llm-entity-resolution
        features_path = Path(__file__).resolve().parent.parent / features_csv

    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)

    lower = gz_config.get('lower_threshold', 4.0)
    upper = gz_config.get('upper_threshold', 6.0)

    gray_zone_df = features_df[
        (features_df['total_score'] >= lower) &
        (features_df['total_score'] < upper)
    ]
    logger.info(f"Found {len(gray_zone_df)} gray zone pairs (score {lower}-{upper})")

    # Load medical records
    logger.info("Loading medical records...")
    medical_records = load_medical_records(str(run_dir))

    # Build record_id → (patient_uuid, facility_id) mapping
    record_map = {}
    for _, row in patients_df.iterrows():
        record_map[row['record_id']] = (row['Id'], row['facility_id'])

    # Generate examples
    examples = []
    for _, row in gray_zone_df.iterrows():
        rid1, rid2 = row['record_id_1'], row['record_id_2']

        if rid1 not in record_map or rid2 not in record_map:
            continue

        uuid1, fac1 = record_map[rid1]
        uuid2, fac2 = record_map[rid2]

        summary_a = summarize_patient_records(uuid1, fac1, medical_records)
        summary_b = summarize_patient_records(uuid2, fac2, medical_records)

        pair = tuple(sorted([rid1, rid2]))
        is_match = pair in true_pairs

        example = dspy.Example(
            medical_history_a=summary_a,
            medical_history_b=summary_b,
            is_match=is_match,
        ).with_inputs('medical_history_a', 'medical_history_b')

        examples.append(example)

    logger.info(f"Built {len(examples)} training examples "
                f"({sum(1 for e in examples if e.is_match)} matches, "
                f"{sum(1 for e in examples if not e.is_match)} non-matches)")

    return examples


def run_optimization(config: dict):
    """
    Run MIPROv2 optimization on the MedicalRecordMatcher.

    Args:
        config: Configuration dictionary
    """
    # Load .env file (for API keys)
    load_dotenv()

    # Configure DSPy task LM
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

    task_lm = dspy.LM(
        model=f"openai/{model_config['name']}",
        api_base=api_base,
        api_key=api_key,
        temperature=model_config.get('temperature', 0.1),
        max_tokens=model_config.get('max_tokens', 256),
    )
    dspy.configure(lm=task_lm)

    # Configure prompt model (stronger LM for instruction generation)
    dspy_config = config.get('dspy', {})
    prompt_model_config = dspy_config.get('prompt_model')
    prompt_lm = None
    if prompt_model_config:
        api_key_env = prompt_model_config.get('api_key_env', '')
        api_key = os.environ.get(api_key_env, '')
        if api_key:
            prompt_lm = dspy.LM(
                model=prompt_model_config['provider'],
                api_key=api_key,
            )
            logger.info(f"Using prompt model: {prompt_model_config['provider']}")
        else:
            logger.warning(f"${api_key_env} not set — falling back to task model for prompt generation")

    # Build training data
    examples = build_training_data(config)

    if len(examples) < 10:
        logger.warning(f"Only {len(examples)} examples — optimization may be unreliable")

    # Split train/val (75/25)
    split_idx = int(len(examples) * 0.75)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]

    logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

    # Create module
    matcher = MedicalRecordMatcher()

    # Run MIPROv2
    optimizer_kwargs = dict(
        metric=accuracy_metric,
        auto=dspy_config.get('auto', 'medium'),
        num_threads=1,  # Serial for local Ollama
    )
    if prompt_lm:
        optimizer_kwargs['prompt_model'] = prompt_lm
        optimizer_kwargs['task_model'] = task_lm
    optimizer = dspy.MIPROv2(**optimizer_kwargs)

    optimized = optimizer.compile(
        matcher,
        trainset=trainset,
        max_bootstrapped_demos=dspy_config.get('max_bootstrapped_demos', 3),
        minibatch_size=dspy_config.get('minibatch_size', 25),
    )

    # Evaluate on validation set
    val_correct = sum(
        1 for ex in valset
        if accuracy_metric(ex, optimized(
            medical_history_a=ex.medical_history_a,
            medical_history_b=ex.medical_history_b,
        ))
    )
    val_acc = val_correct / len(valset) if valset else 0
    logger.info(f"Validation accuracy: {val_acc:.3f} ({val_correct}/{len(valset)})")

    # Save optimized program
    output_path = Path(__file__).resolve().parent.parent / dspy_config.get(
        'optimized_program', 'data/dspy/optimized_program.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(output_path))
    logger.info(f"Saved optimized program to {output_path}")

    return optimized


def main():
    parser = argparse.ArgumentParser(description="Optimize MedicalRecordMatcher with MIPROv2")
    parser.add_argument('--config', default='config/llm_config.yaml',
                        help='Path to configuration YAML')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    config = load_config(args.config)
    run_optimization(config)


if __name__ == '__main__':
    main()
