#!/usr/bin/env python3
"""
Train text-only MedGemma 4B as a binary classifier for entity resolution on GPU.

Uses QLoRA (4-bit NF4) on the text-only base model (vision tower already stripped
by prepare_base_model.py). Trains a 2-class classification head
(Gemma3TextForSequenceClassification) with LoRA on attention + MLP projections.

Usage (RunPod A4000):
    python train_classifier_on_gpu.py
    python train_classifier_on_gpu.py --batch-size 2 --grad-accum 8 --max-length 2048 --gradient-checkpointing
    python train_classifier_on_gpu.py --no-push  # train without pushing adapter
"""

import argparse
import json
import os
from datetime import datetime, timezone

import mlflow
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "abicyclerider/medgemma-4b-text-only-base"


def _flash_attn_available() -> bool:
    """Check if Flash Attention 2 is installed and usable."""
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


DATASET_REPO = "abicyclerider/entity-resolution-pairs"
ADAPTER_REPO = "abicyclerider/medgemma-4b-entity-resolution-classifier"
CHECKPOINT_REPO = "abicyclerider/medgemma-4b-er-classifier-checkpoints"


def compute_metrics(eval_pred):
    """Compute classification metrics for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train MedGemma 4B classifier for entity resolution (QLoRA)"
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (saves VRAM, slower)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Limit training samples (0=use all)"
    )
    parser.add_argument(
        "--no-push", action="store_true", help="Skip pushing adapter to Hub"
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile (inductor backend)",
    )
    args = parser.parse_args()

    # Device detection — QLoRA requires CUDA
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for QLoRA training. Run this on a GPU machine (e.g. RunPod A4000)."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load dataset from Hub and convert to classification format
    print(f"\nLoading dataset from {DATASET_REPO}...")
    dataset = load_dataset(DATASET_REPO)
    print(
        f"  Train: {len(dataset['train'])}, Eval: {len(dataset['eval'])}, "
        f"Test: {len(dataset['test'])}"
    )

    def to_classification(example):
        return {
            "text": example["messages"][0]["content"],
            "label": 1 if example["messages"][1]["content"] == "True" else 0,
        }

    dataset = dataset.map(to_classification)

    # Subsample training data if requested (balanced: equal positive/negative)
    if args.max_samples > 0 and len(dataset["train"]) > args.max_samples:
        print(
            f"  Subsampling train set from {len(dataset['train'])} to {args.max_samples}..."
        )
        train_df = dataset["train"].to_pandas()
        half = args.max_samples // 2
        pos = train_df[train_df["label"] == 1].sample(n=half, random_state=42)
        neg = train_df[train_df["label"] == 0].sample(n=half, random_state=42)
        import pandas as pd

        sampled = (
            pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        )
        from datasets import Dataset

        dataset["train"] = Dataset.from_pandas(sampled)

    print(
        f"  Train label distribution: {sum(dataset['train']['label'])} positive, "
        f"{len(dataset['train']) - sum(dataset['train']['label'])} negative"
    )

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text", "messages"])

    # Load model with 4-bit quantization
    print(f"\nLoading {MODEL_ID} with 4-bit QLoRA quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    load_kwargs = {}
    if _flash_attn_available():
        print("Flash Attention 2 enabled.")
        load_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        print("Flash Attention 2 not installed — using default attention.")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        **load_kwargs,
    )
    # Ensure pad_token_id is set on model config
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model loaded. Parameters: {model.num_parameters():,}")

    # Apply LoRA — target attention projections, save classification head
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    eff_batch = args.batch_size * args.grad_accum
    output_dir = "./output/medgemma-classifier"

    # Configure MLflow tracking (SQLite backend — file store is deprecated in MLflow 3.x)
    mlflow_db = os.path.abspath("./mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("entity-resolution-classifier")

    ta_kwargs = {}
    if args.torch_compile:
        print("torch.compile enabled (inductor backend).")
        ta_kwargs["torch_compile"] = True
        ta_kwargs["torch_compile_backend"] = "inductor"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=True,
        seed=42,
        report_to="mlflow",
        push_to_hub=not args.no_push,
        hub_model_id=CHECKPOINT_REPO,
        hub_strategy="every_save",
        hub_private_repo=True,
        **ta_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    steps_per_epoch = len(dataset["train"]) // eff_batch
    print("\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(
        f"  Batch size: {args.batch_size} x {args.grad_accum} grad_accum = {eff_batch} effective"
    )
    print(f"  Steps/epoch: ~{steps_per_epoch}, Total: ~{steps_per_epoch * args.epochs}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print("  Best model metric: F1")

    # Train and evaluate in a single MLflow run so all metrics appear together
    print("\nStarting training...")
    with mlflow.start_run():
        train_result = trainer.train()

        print("\nTraining complete!")
        print(f"  Total steps: {train_result.global_step}")
        print(f"  Final training loss: {train_result.training_loss:.4f}")

        # Final eval on eval set (logged to the same MLflow run)
        print("\nFinal evaluation on eval set:")
        eval_metrics = trainer.evaluate()
        for k, v in sorted(eval_metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    # Save adapter locally
    adapter_path = f"{output_dir}/best-adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to: {adapter_path}")

    # Save training_metrics.json (gets pushed to HF Hub with the adapter)
    metrics_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hf_repo": ADAPTER_REPO,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eval_f1": eval_metrics.get("eval_f1", 0),
        "eval_precision": eval_metrics.get("eval_precision", 0),
        "eval_recall": eval_metrics.get("eval_recall", 0),
        "training_loss": train_result.training_loss,
        "global_steps": train_result.global_step,
    }
    metrics_path = f"{adapter_path}/training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_info, f, indent=2)
    print(f"Training metrics saved to: {metrics_path}")

    # Push to Hub
    if not args.no_push:
        from huggingface_hub import upload_file

        print(f"\nPushing adapter to {ADAPTER_REPO} (private)...")
        model.push_to_hub(ADAPTER_REPO, private=True)
        tokenizer.push_to_hub(ADAPTER_REPO, private=True)
        upload_file(
            path_or_fileobj=metrics_path,
            path_in_repo="training_metrics.json",
            repo_id=ADAPTER_REPO,
        )

        # Upload MLflow database so it can be downloaded locally
        if os.path.exists(mlflow_db):
            print("Uploading MLflow database to HF Hub...")
            upload_file(
                path_or_fileobj=mlflow_db,
                path_in_repo="mlflow.db",
                repo_id=ADAPTER_REPO,
            )

        print("Done! Adapter + metrics + MLflow logs available on HF Hub.")
    else:
        print("\n--no-push specified, skipping upload.")

    stop_runpod_pod()


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


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        stop_runpod_pod()
        raise SystemExit(1)
