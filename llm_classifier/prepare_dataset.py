#!/usr/bin/env python3
"""
Prepare entity resolution dataset and push to HuggingFace Hub.

Runs locally — loads Synthea data via shared/ modules, generates balanced
train/eval/test pairs with chat-formatted messages, and pushes to HF Hub.

Usage:
    python prepare_dataset.py --augmented-dir output/training/augmented
    python prepare_dataset.py --augmented-dir output/training/augmented --output-dir output/training/dataset
    python prepare_dataset.py --max-length 2048  # filter long pairs (for local training)
    python prepare_dataset.py --no-push           # build dataset without pushing
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login, whoami
from transformers import AutoTokenizer

from shared.data_loader import load_facility_patients
from shared.ground_truth import (
    add_record_ids_to_ground_truth,
    generate_true_pairs_from_ground_truth,
    load_ground_truth,
)
from shared.medical_records import load_medical_records
from shared.summarize import INSTRUCTION, summarize_diff_friendly_from_records

# Only load the record types the summarizer actually uses
SUMMARIZER_RECORD_TYPES = ["conditions", "medications", "allergies", "observations", "procedures"]

MODEL_ID = "google/gemma-3-1b-it"
DATASET_REPO = "abicyclerider/entity-resolution-pairs"


def format_pair(summary_a, summary_b, label):
    """Format a pair as chat messages for SFTTrainer."""
    prompt = INSTRUCTION.format(summary_a=summary_a, summary_b=summary_b)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "True" if label else "False"},
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare entity resolution dataset for HF Hub"
    )
    parser.add_argument(
        "--augmented-dir",
        type=str,
        default=None,
        help="Path to augmented output directory (default: output/augmented)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write dataset_info.json sentinel",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Filter pairs exceeding this token length (0 = no filter)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Build dataset locally without pushing to Hub",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve augmented directory (contains facilities/, metadata/, statistics/)
    _project_root = Path(__file__).resolve().parent.parent
    augmented_dir = args.augmented_dir or str(_project_root / "output" / "augmented")
    if not (Path(augmented_dir) / "facilities").is_dir():
        raise FileNotFoundError(f"No facilities/ directory found in {augmented_dir}")
    RUN_DIR = str(augmented_dir)
    print(f"Using augmented data: {RUN_DIR}")

    # HF login
    if not args.no_push:
        load_dotenv()
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Set it in .env or environment.")
        login(token=token)
        print(f"Logged in as: {whoami()['name']}")

    # Load tokenizer for token stats
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load data
    print(f"Loading data from {RUN_DIR}...")
    patients_df = load_facility_patients(RUN_DIR)
    patients_df["record_id"] = (
        patients_df["facility_id"] + "_" + patients_df["id"].astype(str)
    )

    ground_truth_df = load_ground_truth(RUN_DIR)
    ground_truth_df = add_record_ids_to_ground_truth(ground_truth_df, patients_df)

    true_pairs = generate_true_pairs_from_ground_truth(ground_truth_df)

    # Build record_id -> (patient_uuid, facility_id) mapping
    record_map = {}
    for _, row in patients_df.iterrows():
        record_map[row["record_id"]] = (row["id"], row["facility_id"])

    # Generate non-match pairs (balanced)
    rid_to_true_id = (
        ground_truth_df.dropna(subset=["record_id"])
        .set_index("record_id")["true_patient_id"]
        .to_dict()
    )
    all_record_ids = list(rid_to_true_id.keys())

    random.seed(args.seed)
    non_match_pairs = set()
    target = len(true_pairs)
    attempts = 0
    while len(non_match_pairs) < target and attempts < target * 20:
        r1, r2 = random.sample(all_record_ids, 2)
        if rid_to_true_id.get(r1) != rid_to_true_id.get(r2):
            non_match_pairs.add(tuple(sorted([r1, r2])))
        attempts += 1

    print(f"\nTrue match pairs: {len(true_pairs)}")
    print(f"Non-match pairs:  {len(non_match_pairs)}")

    # Load medical records and build summaries
    print("\nLoading medical records...")
    medical_records = load_medical_records(RUN_DIR, record_types=SUMMARIZER_RECORD_TYPES)

    # Pre-index medical records by (PATIENT, facility_id) for O(1) lookups
    print("Indexing medical records...")
    indexed_records = {}
    for record_type, df in medical_records.items():
        indexed_records[record_type] = {
            key: group for key, group in df.groupby(["PATIENT", "facility_id"])
        }
    # Free the unindexed copies
    del medical_records

    def get_patient_records_indexed(patient_id, facility_id):
        result = {}
        for record_type, idx in indexed_records.items():
            group = idx.get((patient_id, facility_id))
            if group is not None and not group.empty:
                result[record_type] = group
        return result

    all_needed = set()
    for r1, r2 in true_pairs | non_match_pairs:
        all_needed.add(r1)
        all_needed.add(r2)

    print(f"Generating summaries for {len(all_needed)} unique records...")
    summary_cache = {}
    done = 0
    for rid in all_needed:
        if rid in record_map:
            pid, fid = record_map[rid]
            records = get_patient_records_indexed(pid, fid)
            summary_cache[rid] = summarize_diff_friendly_from_records(records)
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(all_needed)} summaries generated...")

    # Token length stats
    token_lengths = [len(tokenizer.encode(s)) for s in summary_cache.values()]
    print("\nSingle summary token lengths:")
    print(
        f"  Mean: {np.mean(token_lengths):.0f}, Median: {np.median(token_lengths):.0f}"
    )
    print(f"  Min: {min(token_lengths)}, Max: {max(token_lengths)}")
    print(f"  95th pctl: {np.percentile(token_lengths, 95):.0f}")

    # Build all pairs
    all_pairs = [(r1, r2, True) for r1, r2 in true_pairs] + [
        (r1, r2, False) for r1, r2 in non_match_pairs
    ]
    random.shuffle(all_pairs)

    # Filter to pairs with summaries
    all_pairs = [
        (r1, r2, l)
        for r1, r2, l in all_pairs
        if r1 in summary_cache and r2 in summary_cache
    ]

    # Optional max_length filter
    if args.max_length > 0:

        def pair_seq_length(r1, r2, label):
            prompt = INSTRUCTION.format(
                summary_a=summary_cache[r1], summary_b=summary_cache[r2]
            )
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "True" if label else "False"},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            return len(tokenizer.encode(text))

        n_before = len(all_pairs)
        all_pairs = [
            (r1, r2, l)
            for r1, r2, l in all_pairs
            if pair_seq_length(r1, r2, l) <= args.max_length
        ]
        n_dropped = n_before - len(all_pairs)
        print(
            f"\nDropped {n_dropped}/{n_before} pairs exceeding {args.max_length} tokens "
            f"({n_dropped / n_before * 100:.1f}%)"
        )

    matches = [p for p in all_pairs if p[2]]
    non_matches = [p for p in all_pairs if not p[2]]

    # Split: 70% train, 15% eval, 15% test (balanced)
    n_total = min(len(matches), len(non_matches))
    n_train = int(n_total * 0.70)
    n_eval = int(n_total * 0.15)
    n_test = n_total - n_train - n_eval

    train_pairs = matches[:n_train] + non_matches[:n_train]
    eval_pairs = (
        matches[n_train : n_train + n_eval] + non_matches[n_train : n_train + n_eval]
    )
    test_pairs = (
        matches[n_train + n_eval : n_train + n_eval + n_test]
        + non_matches[n_train + n_eval : n_train + n_eval + n_test]
    )

    random.shuffle(train_pairs)
    random.shuffle(eval_pairs)
    random.shuffle(test_pairs)

    # Format as chat messages
    def build_split(pairs):
        return [
            format_pair(summary_cache[r1], summary_cache[r2], l) for r1, r2, l in pairs
        ]

    train_data = build_split(train_pairs)
    eval_data = build_split(eval_pairs)
    test_data = build_split(test_pairs)

    print("\nSplits:")
    print(
        f"  Train: {len(train_data)} ({sum(1 for _, _, l in train_pairs if l)} match + "
        f"{sum(1 for _, _, l in train_pairs if not l)} non-match)"
    )
    print(
        f"  Eval:  {len(eval_data)} ({sum(1 for _, _, l in eval_pairs if l)} match + "
        f"{sum(1 for _, _, l in eval_pairs if not l)} non-match)"
    )
    print(
        f"  Test:  {len(test_data)} ({sum(1 for _, _, l in test_pairs if l)} match + "
        f"{sum(1 for _, _, l in test_pairs if not l)} non-match)"
    )

    # Pair token length stats
    pair_lengths = []
    for ex in train_data + eval_data + test_data:
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        pair_lengths.append(len(tokenizer.encode(text)))

    print("\nPair sequence lengths (all splits, with chat template):")
    print(f"  Mean: {np.mean(pair_lengths):.0f}, Median: {np.median(pair_lengths):.0f}")
    print(f"  Min: {min(pair_lengths)}, Max: {max(pair_lengths)}")
    print(f"  95th pctl: {np.percentile(pair_lengths, 95):.0f}")

    # Build HF dataset
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_data),
            "eval": Dataset.from_list(eval_data),
            "test": Dataset.from_list(test_data),
        }
    )

    print(f"\nDataset: {dataset}")

    if not args.no_push:
        print(f"\nPushing to {DATASET_REPO} (private)...")
        dataset.push_to_hub(DATASET_REPO, private=True)
        print("Done! Dataset available on HF Hub.")
    else:
        print("\n--no-push specified, skipping upload.")

    # Write dataset_info.json sentinel
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hf_repo": DATASET_REPO,
            "splits": {
                "train": len(train_data),
                "eval": len(eval_data),
                "test": len(test_data),
            },
            "augmented_dir": str(augmented_dir),
            "seed": args.seed,
        }
        info_path = os.path.join(args.output_dir, "dataset_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\nDataset info written to {info_path}")


if __name__ == "__main__":
    main()
