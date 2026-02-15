#!/usr/bin/env python3
"""
Batch inference for the fine-tuned MedGemma 4B entity resolution classifier.

Loads the merged text-only model (Gemma3TextForSequenceClassification) with
optional 4-bit NF4 quantization and runs batched inference.

Three modes:
  --dataset      Evaluate on the HF test split, print metrics + save predictions
  --input-file   Classify a local file (Parquet or CSV, must have a 'text' column)
  --hf-input     Load input from a HF Hub dataset (use with --hf-output)

Usage:
    # Evaluate on HF test split
    python inference_classifier.py --dataset

    # Classify local file (detects format from extension)
    python inference_classifier.py --input-file gray_zone.parquet --output-file predictions.parquet

    # HF Hub round-trip (for remote GPU inference on RunPod / Vertex AI)
    # Note: HF Hub repos use "grey" spelling (external resource, can't rename)
    python inference_classifier.py \
        --hf-input abicyclerider/grey-zone-pairs \
        --hf-output abicyclerider/grey-zone-predictions

    # Use bf16 instead of 4-bit quantization
    python inference_classifier.py --dataset --no-quantize
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

MODEL_ID = "abicyclerider/medgemma-4b-entity-resolution-text-only"
DATASET_REPO = "abicyclerider/entity-resolution-pairs"


def load_model(model_id=MODEL_ID, quantize_4bit=True, device="cuda"):
    """Load the merged text-only classifier. Returns (model, tokenizer)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if quantize_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        print(f"Loading {model_id} with 4-bit NF4 quantization...")
        load_kwargs["device_map"] = "auto"
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif device == "cuda":
        print(f"Loading {model_id} in bf16 (no quantization)...")
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        # MPS or CPU — load directly to device to avoid double-buffering
        print(f"Loading {model_id} in float16 on {device}...")
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = {"": device}

    model = AutoModelForSequenceClassification.from_pretrained(model_id, **load_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready on {device}. Parameters: {total_params:,}")
    return model, tokenizer


def predict_batch(model, tokenizer, texts, max_length=2048):
    """Run inference on a batch of texts. Returns (predictions, confidences)."""
    inputs = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits.float(), dim=-1)
    predictions = probs.argmax(dim=-1).cpu().numpy()
    confidences = probs.max(dim=-1).values.cpu().numpy()
    return predictions, confidences


def evaluate_test_split(model, tokenizer, batch_size, max_length, output_csv):
    """Load HF test split, run batched inference, compute & print metrics."""
    from datasets import load_dataset

    print(f"\nLoading test split from {DATASET_REPO}...")
    dataset = load_dataset(DATASET_REPO, split="test")
    print(f"  Test examples: {len(dataset)}")

    # Extract text and labels using the same format as training
    texts = [ex["messages"][0]["content"] for ex in dataset]
    labels = [1 if ex["messages"][1]["content"] == "True" else 0 for ex in dataset]
    labels = np.array(labels)

    print(
        f"  Label distribution: {labels.sum()} positive, {len(labels) - labels.sum()} negative"
    )

    # Batched inference
    all_preds = []
    all_confs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"\nRunning inference ({n_batches} batches, batch_size={batch_size})...")
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        preds, confs = predict_batch(model, tokenizer, batch_texts, max_length)
        all_preds.extend(preds)
        all_confs.extend(confs)

        batch_num = i // batch_size + 1
        if batch_num % 10 == 0 or batch_num == n_batches:
            elapsed = time.time() - t0
            rate = (i + len(batch_texts)) / elapsed
            print(f"  Batch {batch_num}/{n_batches} — {rate:.1f} examples/sec")

    elapsed = time.time() - t0
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)

    # Metrics
    print(f"\n{'=' * 60}")
    print(f"Test Set Results ({len(texts)} examples, {elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {accuracy_score(labels, all_preds):.4f}")
    print(f"  Precision: {precision_score(labels, all_preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(labels, all_preds, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(labels, all_preds, zero_division=0):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            labels, all_preds, target_names=["Non-match (0)", "Match (1)"]
        )
    )

    # Save predictions
    df = pd.DataFrame(
        {
            "text": [t[:200] + "..." if len(t) > 200 else t for t in texts],
            "label": labels,
            "prediction": all_preds,
            "confidence": np.round(all_confs, 4),
            "correct": (labels == all_preds).astype(int),
        }
    )
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    # Error analysis summary
    errors = df[df["correct"] == 0]
    if len(errors) > 0:
        print(
            f"\nError summary: {len(errors)} misclassified ({len(errors) / len(df) * 100:.1f}%)"
        )
        fp = ((all_preds == 1) & (labels == 0)).sum()
        fn = ((all_preds == 0) & (labels == 1)).sum()
        print(f"  False positives: {fp}, False negatives: {fn}")
        print(f"  Avg confidence on errors: {errors['confidence'].mean():.4f}")
        print(
            f"  Avg confidence on correct: {df[df['correct'] == 1]['confidence'].mean():.4f}"
        )


def _read_df(path: str) -> pd.DataFrame:
    """Read a DataFrame from CSV or Parquet based on file extension."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_df(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV or Parquet based on file extension."""
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def classify_file(model, tokenizer, input_path, output_path, batch_size, max_length):
    """Load an input file (CSV or Parquet), run batched inference, save results."""
    print(f"\nLoading input: {input_path}")
    df = _read_df(input_path)

    if "text" not in df.columns:
        print(
            f"ERROR: Input must have a 'text' column. Found columns: {list(df.columns)}"
        )
        sys.exit(1)

    texts = df["text"].tolist()
    has_labels = "label" in df.columns
    print(f"  Loaded {len(texts)} examples" + (" (with labels)" if has_labels else ""))

    # Batched inference
    all_preds = []
    all_confs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"\nRunning inference ({n_batches} batches, batch_size={batch_size})...")
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        preds, confs = predict_batch(model, tokenizer, batch_texts, max_length)
        all_preds.extend(preds)
        all_confs.extend(confs)

        batch_num = i // batch_size + 1
        if batch_num % 10 == 0 or batch_num == n_batches:
            elapsed = time.time() - t0
            rate = (i + len(batch_texts)) / elapsed
            print(f"  Batch {batch_num}/{n_batches} — {rate:.1f} examples/sec")

    elapsed = time.time() - t0
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)

    # Build output — drop text from result
    result = df.drop(columns=["text"], errors="ignore").copy()
    result["prediction"] = all_preds
    result["confidence"] = np.round(all_confs, 4)

    if has_labels:
        labels = df["label"].values
        result["correct"] = (labels == all_preds).astype(int)
        print(f"\n  Accuracy: {accuracy_score(labels, all_preds):.4f}")
        print(f"  F1:       {f1_score(labels, all_preds, zero_division=0):.4f}")

    _write_df(result, output_path)
    print(f"\n{len(texts)} predictions saved to {output_path} ({elapsed:.1f}s)")
    print(
        f"  Predicted match: {(all_preds == 1).sum()}, non-match: {(all_preds == 0).sum()}"
    )


def classify_hf_dataset(model, tokenizer, hf_input, hf_output, batch_size, max_length):
    """Load a dataset from HF Hub, run batched inference, push results back."""
    from datasets import Dataset, load_dataset

    print(f"\nLoading dataset from HF Hub: {hf_input}")
    ds = load_dataset(hf_input, split="train")
    df = ds.to_pandas()

    if "text" not in df.columns:
        print(
            f"ERROR: Dataset must have a 'text' column. Found columns: {list(df.columns)}"
        )
        sys.exit(1)

    texts = df["text"].tolist()
    has_labels = "label" in df.columns
    print(f"  Loaded {len(texts)} examples" + (" (with labels)" if has_labels else ""))

    # Batched inference
    all_preds = []
    all_confs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"\nRunning inference ({n_batches} batches, batch_size={batch_size})...")
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        preds, confs = predict_batch(model, tokenizer, batch_texts, max_length)
        all_preds.extend(preds)
        all_confs.extend(confs)

        batch_num = i // batch_size + 1
        if batch_num % 10 == 0 or batch_num == n_batches:
            elapsed = time.time() - t0
            rate = (i + len(batch_texts)) / elapsed
            print(f"  Batch {batch_num}/{n_batches} — {rate:.1f} examples/sec")

    elapsed = time.time() - t0
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)

    # Build output — preserve input columns (except text), add prediction + confidence
    result = df.drop(columns=["text"], errors="ignore").copy()
    result["prediction"] = all_preds
    result["confidence"] = np.round(all_confs, 4)

    if has_labels:
        labels = df["label"].values
        result["correct"] = (labels == all_preds).astype(int)
        print(f"\n  Accuracy: {accuracy_score(labels, all_preds):.4f}")
        print(f"  F1:       {f1_score(labels, all_preds, zero_division=0):.4f}")

    print(f"\n{len(texts)} predictions completed ({elapsed:.1f}s)")
    print(
        f"  Predicted match: {(all_preds == 1).sum()}, non-match: {(all_preds == 0).sum()}"
    )

    # Push to HF Hub
    if hf_output:
        print(f"\nPushing predictions to HF Hub: {hf_output}")
        Dataset.from_pandas(result).push_to_hub(hf_output)
        print(f"  Done — {len(result)} rows pushed to {hf_output}")
    else:
        # Fall back to local CSV
        output_csv = "predictions.csv"
        result.to_csv(output_csv, index=False)
        print(f"  Saved to {output_csv}")


def stop_runpod_pod():
    """Stop the current RunPod pod via API. No-op when not on RunPod."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not api_key:
        return
    try:
        import requests

        print(f"\nStopping RunPod pod {pod_id}...")
        resp = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": f'mutation {{ podStop(input: {{podId: "{pod_id}"}}) {{ id }} }}'
            },
            timeout=30,
        )
        resp.raise_for_status()
        print("  Pod stop requested.")
    except Exception as e:
        print(f"  Warning: failed to stop pod: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for MedGemma 4B entity resolution classifier"
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dataset",
        action="store_true",
        help="Evaluate on the HF test split and print metrics",
    )
    mode.add_argument(
        "--input-file",
        type=str,
        help="Path to input file (Parquet or CSV) with a 'text' column",
    )
    mode.add_argument(
        "--hf-input",
        type=str,
        help="HF Hub dataset repo to load as input (e.g. abicyclerider/grey-zone-pairs)",  # "grey" spelling in HF repo name
    )

    # Options
    parser.add_argument(
        "--output-file",
        type=str,
        default="predictions.parquet",
        help="Path for output predictions (default: predictions.parquet)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Inference batch size (default: 16)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length for tokenization (default: 2048)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Load in bf16 instead of 4-bit NF4 (needs more VRAM)",
    )
    parser.add_argument(
        "--hf-output",
        type=str,
        default=None,
        help="HF Hub repo to push predictions to (requires --hf-input)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=f"HF model repo (default: {MODEL_ID})",
    )
    args = parser.parse_args()

    if args.hf_output and not args.hf_input:
        parser.error("--hf-output requires --hf-input")

    # Detect compute device: CUDA > MPS > fail
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Device: Apple Silicon (MPS)")
        if not args.no_quantize:
            print("  4-bit quantization requires CUDA — using float16 instead")
            args.no_quantize = True
        if args.batch_size > 2:
            print(f"  Reducing batch size from {args.batch_size} to 2 for MPS memory")
            args.batch_size = 2
    else:
        print("ERROR: No GPU available. Requires CUDA or Apple Silicon (MPS).")
        sys.exit(1)

    try:
        # Load model
        model, tokenizer = load_model(
            model_id=args.model_id,
            quantize_4bit=not args.no_quantize,
            device=device,
        )

        # Run inference
        if args.dataset:
            evaluate_test_split(
                model, tokenizer, args.batch_size, args.max_length, args.output_file
            )
        elif args.hf_input:
            classify_hf_dataset(
                model,
                tokenizer,
                args.hf_input,
                args.hf_output,
                args.batch_size,
                args.max_length,
            )
        else:
            classify_file(
                model,
                tokenizer,
                args.input_file,
                args.output_file,
                args.batch_size,
                args.max_length,
            )
    finally:
        stop_runpod_pod()


if __name__ == "__main__":
    main()
