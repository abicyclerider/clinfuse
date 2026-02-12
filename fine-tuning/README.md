# Fine-Tuning: MedGemma Entity Resolution Classifier

Fine-tuned MedGemma 4B (text-only) for pairwise patient entity resolution on Synthea synthetic medical records. Uses QLoRA (r=16, attention + MLP targets) with a sequence classification head.

**Performance:** F1 = 0.963, Accuracy = 0.963, Precision = 0.969, Recall = 0.958 (6,482-pair test set)

**HF Hub:**
- Model: [`abicyclerider/medgemma-4b-entity-resolution-text-only`](https://huggingface.co/abicyclerider/medgemma-4b-entity-resolution-text-only) (3.88B params, LoRA merged, no vision tower)
- Dataset: [`abicyclerider/entity-resolution-pairs`](https://huggingface.co/datasets/abicyclerider/entity-resolution-pairs) (30K train / 6.5K eval / 6.5K test, balanced)

## Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Prepare | `prepare_dataset.py` | Load Synthea data, generate Strategy D summaries, build balanced splits, push to HF Hub |
| 2. Train | `train_classifier_on_gpu.py` | QLoRA fine-tuning on GPU (H100 ~2.2h, L40S needs gradient checkpointing) |
| 3. Export | `export_text_only_model.py` | Merge LoRA adapter, strip vision tower, upload merged model |
| 4. Infer | `inference_classifier.py` | Batch inference & evaluation on test set or custom CSV |

## Quick Start

```bash
# 1. Prepare dataset (runs locally, pushes to HF Hub)
python prepare_dataset.py

# 2. Train on GPU (see RUNPOD_GUIDE.md for cloud setup)
python train_classifier_on_gpu.py

# 3. Export merged text-only model
python export_text_only_model.py

# 4. Evaluate on HF test split
python inference_classifier.py --dataset

# 4b. Classify a custom CSV (must have a 'text' column)
python inference_classifier.py --input-csv pairs.csv --output-csv predictions.csv
```

Inference requires a CUDA GPU. Uses 4-bit NF4 quantization by default (`--no-quantize` for bf16).

## Setup

```bash
cd fine-tuning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # training
pip install -r requirements-inference.txt # inference only
```

## Docker (Inference)

```bash
docker build -t medgemma-er .
docker run --gpus all medgemma-er --dataset
```

## GPU Training on RunPod

See [`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md) for provisioning, SSH access, and training commands.
