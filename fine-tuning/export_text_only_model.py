#!/usr/bin/env python3
"""
Merge LoRA adapter into the text-only MedGemma 4B base and export.

Loads the text-only base model (already stripped of vision tower by
prepare_base_model.py), applies the LoRA adapter, merges weights, and saves
the resulting standalone Gemma3TextForSequenceClassification model.

The resulting model:
  - Doesn't require the peft library to load
  - Loads via AutoModelForSequenceClassification.from_pretrained()

Usage (on a GPU with >=16GB VRAM):
    python export_text_only_model.py
    python export_text_only_model.py --validate          # also run test set eval
    python export_text_only_model.py --no-push --local-dir ./merged-model
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import torch

BASE_MODEL_ID = "abicyclerider/medgemma-4b-text-only-base"
ADAPTER_REPO = "abicyclerider/medgemma-4b-entity-resolution-classifier"
OUTPUT_REPO = "abicyclerider/medgemma-4b-entity-resolution-text-only"
DATASET_REPO = "abicyclerider/entity-resolution-pairs"


def merge_adapter(base_model_id, adapter_repo):
    """Load text-only base model + LoRA adapter, merge, return merged model."""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Load tokenizer
    print(f"Loading tokenizer from {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load text-only base model in bf16 (no quantization — need full precision for clean merge)
    print(f"Loading {base_model_id} in bf16...")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Base model type: {type(model).__name__}")
    print(f"Base model params: {sum(p.numel() for p in model.parameters()):,}")

    # Apply LoRA adapter
    print(f"\nApplying LoRA adapter from {adapter_repo}...")
    model = PeftModel.from_pretrained(model, adapter_repo)

    # Merge LoRA into base weights
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()
    print(f"Merged model type: {type(model).__name__}")
    print(f"Merged model params: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def validate_on_test(model, tokenizer, batch_size=16, max_length=2048):
    """Run test set evaluation and print metrics."""
    import time

    import numpy as np
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    print(f"\nLoading test split from {DATASET_REPO}...")
    dataset = load_dataset(DATASET_REPO, split="test")
    texts = [ex["messages"][0]["content"] for ex in dataset]
    labels = np.array(
        [1 if ex["messages"][1]["content"] == "True" else 0 for ex in dataset]
    )
    print(f"  Test examples: {len(texts)}")
    print(
        f"  Label distribution: {labels.sum()} positive, {len(labels) - labels.sum()} negative"
    )

    all_preds = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"\nRunning inference ({n_batches} batches, batch_size={batch_size})...")
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)

        batch_num = i // batch_size + 1
        if batch_num % 50 == 0 or batch_num == n_batches:
            elapsed = time.time() - t0
            rate = (i + len(batch_texts)) / elapsed
            print(f"  Batch {batch_num}/{n_batches} -- {rate:.1f} examples/sec")

    elapsed = time.time() - t0
    all_preds = np.array(all_preds)

    acc = accuracy_score(labels, all_preds)
    prec = precision_score(labels, all_preds, zero_division=0)
    rec = recall_score(labels, all_preds, zero_division=0)
    f1 = f1_score(labels, all_preds, zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"Merged Model — Test Set Results ({len(texts)} examples, {elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")

    if abs(f1 - 0.963) > 0.005:
        print(
            f"\n  WARNING: F1 {f1:.4f} differs from expected 0.963 by more than 0.005!"
        )
    else:
        print("\n  OK: F1 matches expected 0.963 (within tolerance)")

    return {
        "test_f1": f1,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into text-only MedGemma 4B base"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run test set evaluation after export"
    )
    parser.add_argument("--no-push", action="store_true", help="Skip pushing to HF Hub")
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./merged-text-only",
        help="Local directory to save the merged model",
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        default=OUTPUT_REPO,
        help=f"HF Hub repo to push to (default: {OUTPUT_REPO})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for validation (default: 16)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for bf16 model loading.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Merge adapter
    model, tokenizer = merge_adapter(BASE_MODEL_ID, ADAPTER_REPO)

    # Validate
    test_metrics = {}
    if args.validate:
        test_metrics = validate_on_test(model, tokenizer, batch_size=args.batch_size)

    # Save locally
    print(f"\nSaving merged model to {args.local_dir}...")
    model.save_pretrained(args.local_dir)
    tokenizer.save_pretrained(args.local_dir)
    print("Saved.")

    # Save export_info.json (gets pushed to HF Hub with the model)
    total_params = sum(p.numel() for p in model.parameters())
    export_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hf_repo": args.output_repo,
        "base_model": BASE_MODEL_ID,
        "adapter_repo": ADAPTER_REPO,
        "params": total_params,
        **test_metrics,
    }
    export_info_path = f"{args.local_dir}/export_info.json"
    with open(export_info_path, "w") as f:
        json.dump(export_info, f, indent=2)
    print(f"Export info saved to: {export_info_path}")

    # Verify it loads cleanly via Auto class
    print("\nVerifying Auto class loading...")
    from transformers import AutoModelForSequenceClassification

    reloaded = AutoModelForSequenceClassification.from_pretrained(
        args.local_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Reloaded type: {type(reloaded).__name__}")
    print(f"  Reloaded params: {sum(p.numel() for p in reloaded.parameters()):,}")
    del reloaded
    torch.cuda.empty_cache()

    # Push to Hub
    if not args.no_push:
        from huggingface_hub import upload_file

        print(f"\nPushing to {args.output_repo} (private)...")
        model.push_to_hub(args.output_repo, private=True)
        tokenizer.push_to_hub(args.output_repo, private=True)
        upload_file(
            path_or_fileobj=export_info_path,
            path_in_repo="export_info.json",
            repo_id=args.output_repo,
        )
        print("Done! Model + export info available on HF Hub.")
    else:
        print("\n--no-push specified, skipping upload.")


if __name__ == "__main__":
    main()
