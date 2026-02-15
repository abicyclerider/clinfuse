# SyntheticMass

Entity resolution pipeline for synthetic medical records, built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge).

## Overview

Generate synthetic patient data, augment it with realistic errors and duplicates, then resolve matching records using rule-based blocking and a fine-tuned MedGemma classifier. The fine-tuned model achieves **F1 = 0.963** on held-out test pairs.

## Project Structure

```
SyntheticMass/
├── synthea_runner/          # Synthea patient data generation (Docker submodule)
├── augmentation/            # Error injection, duplicates, ground truth labels
├── entity_resolution/       # Splink blocking, pair generation, golden records
│   └── core/                # Core algorithms (splink_linker, golden_record, evaluation)
├── fine_tuning/             # MedGemma 4B QLoRA training & inference
├── shared/                  # Shared utilities (data_loader, summarizer, ground_truth)
├── output/                  # DVC-managed pipeline outputs (gitignored)
├── dvc.yaml                 # Pipeline definition (10 stages)
├── params.yaml              # Pipeline parameters
└── pyproject.toml           # Python project config (ruff, mypy, pytest)
```

## DVC Pipeline

The pipeline is managed by [DVC](https://dvc.org/) and defined in `dvc.yaml`. Parameters are in `params.yaml`.

**Inference track** (main pipeline):

```
generate → augment → resolve → infer → golden_records
```

1. **generate** — Run Synthea to create synthetic patient CSVs
2. **augment** — Inject errors, distribute patients across facilities, create ground truth
3. **resolve** — Splink probabilistic linkage: auto-matches + gray zone pairs with LLM text
4. **infer** — MedGemma classifier scores gray zone pairs (RunPod GPU)
5. **golden_records** — Combine auto-matches + LLM predictions into final golden records, evaluate

**Training track** (model fine-tuning):

```
generate_training → augment_training → prepare_dataset → train → export
```

6. **generate_training** — Separate Synthea run (different seed/population)
7. **augment_training** — Augment training data
8. **prepare_dataset** — Build HuggingFace dataset with Strategy D summaries
9. **train** — QLoRA fine-tuning on RunPod GPU
10. **export** — Merge LoRA adapter into base model, push to HF Hub

### Quick Start

```bash
# Install Python dependencies
pip install -e .

# Run the full inference pipeline
dvc repro golden_records

# Run a single stage
dvc repro resolve

# View the pipeline DAG
dvc dag
```

GPU stages (infer, train, export) run on [RunPod](https://www.runpod.io/) via automated scripts. See [`fine_tuning/README.md`](fine_tuning/README.md) for details.

## Key Results

| Model | Params | F1 | Accuracy | Precision | Recall |
|-------|--------|----|----------|-----------|--------|
| MedGemma 4B text-only (QLoRA, merged) | 3.88B | 0.963 | 0.963 | 0.969 | 0.958 |

Model on HuggingFace: [`abicyclerider/medgemma-4b-entity-resolution-text-only`](https://huggingface.co/abicyclerider/medgemma-4b-entity-resolution-text-only)

## Getting Started

### Prerequisites

- **Docker** — for Synthea data generation and entity resolution
- **Python 3.11+** — for augmentation, entity resolution, and dataset preparation
- **GPU (48GB+ VRAM)** — for fine-tuning and inference (H100 or L40S recommended)
- **DVC** — for pipeline orchestration (`pip install dvc`)

### Generate Base Data

```bash
git submodule update --init --recursive
cd synthea_runner
docker compose up
```

See [`synthea_runner/README.md`](synthea_runner/README.md) for configuration details.

## License

See `LICENSE` file for details.
