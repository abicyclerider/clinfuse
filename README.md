# SyntheticMass

Entity resolution pipeline for synthetic medical records, built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge).

## Overview

Generate synthetic patient data, augment it with realistic errors and duplicates, then resolve matching records using rule-based blocking and a fine-tuned MedGemma classifier. The fine-tuned model achieves **F1 = 0.963** on held-out test pairs.

## Project Structure

```
SyntheticMass/
├── synthea-runner/       # Synthea patient data generation (Docker)
├── augmentation/         # Error injection, duplicates, ground truth labels
├── entity_resolution/    # Blocking, candidate pair generation, classification
├── fine-tuning/          # MedGemma 4B QLoRA training & evaluation
├── shared/               # Shared utilities (summarizer, etc.)
├── analysis/             # Exploratory notebooks
└── README.md
```

## Workflow

1. **Generate** — `synthea-runner/` creates 10K synthetic patients via [Synthea](https://github.com/synthetichealth/synthea)
2. **Augment** — `augmentation/` injects typos, formatting changes, missing fields, and duplicates with ground-truth labels
3. **Block & Pair** — `entity_resolution/` applies rule-based blocking to reduce the candidate space, then generates record pairs
4. **Classify** — Fine-tuned MedGemma 4B text-only classifier scores each pair as match/non-match

## Key Results

| Model | Params | F1 | Accuracy | Precision | Recall |
|-------|--------|----|----------|-----------|--------|
| MedGemma 4B text-only (QLoRA, merged) | 3.88B | 0.963 | 0.963 | 0.969 | 0.958 |

Model on HuggingFace: [`abicyclerider/medgemma-4b-entity-resolution-text-only`](https://huggingface.co/abicyclerider/medgemma-4b-entity-resolution-text-only)

## Getting Started

### Prerequisites

- **Docker** — for Synthea data generation
- **Python 3.11+** — for augmentation, entity resolution, and analysis
- **GPU (48GB+ VRAM)** — for fine-tuning (H100 or L40S recommended)

### Generate Base Data

```bash
git submodule update --init --recursive
cd synthea-runner
docker compose up
```

See [`synthea-runner/README.md`](synthea-runner/README.md) for configuration details.

## License

See `LICENSE` file for details.
