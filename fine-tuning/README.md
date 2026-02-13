# Fine-Tuning: MedGemma Entity Resolution Classifier

Fine-tuned MedGemma 4B (text-only) for pairwise patient entity resolution on Synthea synthetic medical records. Uses QLoRA (r=16, attention + MLP targets) with a sequence classification head.

**Performance:** F1 = 0.963, Accuracy = 0.963, Precision = 0.969, Recall = 0.958 (6,482-pair test set)

**HF Hub:**
- Merged model: [`abicyclerider/medgemma-4b-entity-resolution-text-only`](https://huggingface.co/abicyclerider/medgemma-4b-entity-resolution-text-only) (3.88B params, LoRA merged)
- Text-only base: [`abicyclerider/medgemma-4b-text-only-base`](https://huggingface.co/abicyclerider/medgemma-4b-text-only-base) (3.88B params, no fine-tuning)
- Dataset: [`abicyclerider/entity-resolution-pairs`](https://huggingface.co/datasets/abicyclerider/entity-resolution-pairs) (30K train / 6.5K eval / 6.5K test, balanced)

## Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Prepare dataset | `prepare_dataset.py` | Load Synthea data, generate Strategy D summaries, build balanced splits, push to HF Hub |
| 2. Prepare base model | `prepare_base_model.py` | Strip vision tower from `google/medgemma-4b-it`, push text-only base to HF Hub (one-time, CPU) |
| 3. Train | `train_classifier_on_gpu.py` | QLoRA fine-tuning on text-only base (H100 ~2.2h, L40S needs gradient checkpointing) |
| 4. Export | `export_text_only_model.py` | Merge LoRA adapter into base, upload merged model |
| 5. Infer | `inference_classifier.py` | Batch inference & evaluation on test set or custom CSV |

## Quick Start

```bash
# 1. Prepare dataset (runs locally, pushes to HF Hub)
python prepare_dataset.py

# 2. Prepare text-only base model (one-time, runs on CPU)
python prepare_base_model.py

# 3. Train on GPU (see RUNPOD_GUIDE.md for cloud setup)
python train_classifier_on_gpu.py

# 4. Export merged model
python export_text_only_model.py

# 5. Evaluate on HF test split
python inference_classifier.py --dataset

# 5b. Classify a custom CSV (must have a 'text' column)
python inference_classifier.py --input-csv pairs.csv --output-csv predictions.csv
```

Inference requires a CUDA GPU. Uses 4-bit NF4 quantization by default (`--no-quantize` for bf16).

## Docker

```bash
docker build -t medgemma .

# Prepare base model (CPU — one-time)
docker run -e HF_TOKEN=hf_... medgemma python prepare_base_model.py

# Train (GPU)
docker run --gpus all -e HF_TOKEN=hf_... medgemma python train_classifier_on_gpu.py --epochs 3

# Export (GPU)
docker run --gpus all -e HF_TOKEN=hf_... medgemma python export_text_only_model.py --validate

# Infer (GPU)
docker run --gpus all -e HF_TOKEN=hf_... medgemma python inference_classifier.py --dataset
```

## Local Setup (without Docker)

```bash
cd fine-tuning
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## GPU Training on RunPod

See [`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md) for provisioning, SSH access, and training commands.
