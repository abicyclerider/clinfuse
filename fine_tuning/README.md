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
| 3. Train | `train_classifier.py` | QLoRA fine-tuning on text-only base (H100 ~2.2h, L40S needs gradient checkpointing) |
| 4. Export | `export_model.py` | Merge LoRA adapter into base, upload merged model |
| 5. Infer | `infer_classifier.py` | Batch inference & evaluation on test set, custom CSV, or HF Hub dataset |

## Quick Start

```bash
# 1. Prepare dataset (runs locally, pushes to HF Hub)
python prepare_dataset.py

# 2. Prepare text-only base model (one-time, runs on CPU)
python prepare_base_model.py

# 3. Train on GPU (see RUNPOD_GUIDE.md for cloud setup)
python train_classifier.py

# 4. Export merged model
python export_model.py

# 5. Evaluate on HF test split
python infer_classifier.py --dataset

# 5b. Classify a custom CSV (must have a 'text' column)
python infer_classifier.py --input-csv pairs.csv --output-csv predictions.csv

# 5c. HF Hub round-trip (for remote GPU — RunPod, Vertex AI)
python infer_classifier.py \
    --hf-input abicyclerider/grey-zone-pairs \
    --hf-output abicyclerider/grey-zone-predictions
```

Inference requires a CUDA GPU. Uses 4-bit NF4 quantization by default (`--no-quantize` for bf16).

## Docker

```bash
docker build -t medgemma .

# Prepare base model (CPU — one-time)
docker run -e HF_TOKEN=hf_... medgemma python prepare_base_model.py

# Train (GPU)
docker run --gpus all -e HF_TOKEN=hf_... medgemma python train_classifier.py --epochs 3

# Export (GPU)
docker run --gpus all -e HF_TOKEN=hf_... medgemma python export_model.py --validate

# Infer (GPU)
docker run --gpus all -e HF_TOKEN=hf_... medgemma python infer_classifier.py --dataset
```

## Local Setup (without Docker)

```bash
cd fine_tuning
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Remote GPU on RunPod

The `launch_pod.sh` script launches any pipeline stage on a RunPod GPU pod. It uses HF Hub as the data bridge — no SSH or file copying needed.

```bash
# 1. Upload gray zone pairs to HF Hub (note: HF repo uses "grey" spelling)
python -c "
from datasets import Dataset
import pandas as pd
df = pd.read_csv('output/resolved/gray_zone_pairs.csv')
Dataset.from_pandas(df).push_to_hub('abicyclerider/grey-zone-pairs')
"

# 2. Launch inference on RunPod (fire and forget)
cd fine_tuning
./launch_pod.sh infer \
    --hf-input abicyclerider/grey-zone-pairs \
    --hf-output abicyclerider/grey-zone-predictions

# 3. Pod auto-stops when done. Download predictions:
python -c "
from datasets import load_dataset
ds = load_dataset('abicyclerider/grey-zone-predictions', split='train')
ds.to_pandas().to_csv('output/inferred/predictions.csv', index=False)
"

# 4. Continue DVC pipeline
dvc repro golden_records
```

Override GPU type (default: NVIDIA L40S):

```bash
./launch_pod.sh infer --gpu-type "NVIDIA A100 80GB PCIe" \
    --hf-input abicyclerider/grey-zone-pairs \
    --hf-output abicyclerider/grey-zone-predictions
```

Prerequisites:
- `runpodctl` configured (`~/.runpod/config.toml` with API key)
- `HF_TOKEN` in `fine_tuning/.env`
- GHCR package set to public (`gh api -X PUT /user/packages/container/medgemma-pipeline/visibility -f visibility=public`)

See [`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md) for SSH-based provisioning and manual training commands.
