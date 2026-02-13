#!/usr/bin/env python3
"""
Extract text-only base model from multimodal MedGemma 4B.

One-time step that strips the vision tower (SigLIP) and multi-modal projector
from google/medgemma-4b-it, producing a leaner Gemma3TextForSequenceClassification
base model suitable for QLoRA training.

The resulting model:
  - Has no vision tower (~420M fewer params, ~0.8GB less in bf16)
  - Doesn't need token_type_ids
  - Has a randomly initialized 2-class classification head
  - Loads via AutoModelForSequenceClassification.from_pretrained()

Can run on CPU with 18GB RAM (no inference, just state_dict extraction).

Usage:
    python prepare_base_model.py
    python prepare_base_model.py --no-push --local-dir ./text-only-base
"""

import argparse

import torch

SOURCE_MODEL_ID = "google/medgemma-4b-it"
OUTPUT_REPO = "abicyclerider/medgemma-4b-text-only-base"


def extract_text_model(source_model_id):
    """Load multimodal model, extract text backbone + classification head."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Gemma3TextConfig,
        Gemma3TextForSequenceClassification,
    )

    # Load tokenizer
    print(f"Loading tokenizer from {source_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(source_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load multimodal model in bf16 on CPU
    print(f"Loading {source_model_id} in bf16 (CPU)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        source_model_id,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Multimodal model params: {total_params:,}")

    # Extract text backbone weights
    # Multimodal layout: model.model.language_model.* -> Text-only: model.*
    print("\nExtracting text backbone state_dict...")
    lang_state = model.model.language_model.state_dict()
    score_state = model.score.state_dict()
    print(f"  Language model keys: {len(lang_state)}")
    print(f"  Score head keys: {len(score_state)}")

    # Build text-only config from the multimodal model's text_config
    text_config_dict = model.config.text_config.to_dict()
    text_config = Gemma3TextConfig(**text_config_dict)
    text_config.num_labels = 2
    text_config.pad_token_id = tokenizer.pad_token_id

    # Instantiate text-only classifier and load weights
    print("\nCreating Gemma3TextForSequenceClassification...")
    text_model = Gemma3TextForSequenceClassification(text_config)
    text_model = text_model.to(dtype=torch.bfloat16)
    text_model.model.load_state_dict(lang_state)
    text_model.score.load_state_dict(score_state)

    text_params = sum(p.numel() for p in text_model.parameters())
    print(f"Text-only model params: {text_params:,}")
    print(f"  (stripped ~{total_params - text_params:,} params from vision tower)")

    # Free multimodal model
    del model, lang_state, score_state

    return text_model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Extract text-only base model from multimodal MedGemma 4B"
    )
    parser.add_argument("--no-push", action="store_true",
                        help="Skip pushing to HF Hub")
    parser.add_argument("--local-dir", type=str, default="./text-only-base",
                        help="Local directory to save the model")
    parser.add_argument("--output-repo", type=str, default=OUTPUT_REPO,
                        help=f"HF Hub repo to push to (default: {OUTPUT_REPO})")
    args = parser.parse_args()

    text_model, tokenizer = extract_text_model(SOURCE_MODEL_ID)

    # Save locally
    print(f"\nSaving text-only base model to {args.local_dir}...")
    text_model.save_pretrained(args.local_dir)
    tokenizer.save_pretrained(args.local_dir)
    print("Saved.")

    # Verify it loads cleanly via Auto class
    print("\nVerifying Auto class loading...")
    from transformers import AutoModelForSequenceClassification
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        args.local_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(f"  Reloaded type: {type(reloaded).__name__}")
    print(f"  Reloaded params: {sum(p.numel() for p in reloaded.parameters()):,}")
    del reloaded

    # Push to Hub
    if not args.no_push:
        print(f"\nPushing to {args.output_repo} (private)...")
        text_model.push_to_hub(args.output_repo, private=True)
        tokenizer.push_to_hub(args.output_repo, private=True)
        print("Done! Base model available on HF Hub.")
    else:
        print("\n--no-push specified, skipping upload.")


if __name__ == "__main__":
    main()
