#!/usr/bin/env bash
# Automated remote training round-trip:
#   Launch RunPod pod → QLoRA training → adapter pushed to HF Hub → download metrics
#
# Usage: ./train_remote.sh [--gpu-type "GPU NAME"] <output_dir> [-- training_args...]
# Example: ./train_remote.sh output/training/train -- --epochs 3 --batch-size 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_helpers.sh"

ADAPTER_REPO="abicyclerider/medgemma-4b-entity-resolution-classifier"
TIMEOUT=14400  # 4 hours
STALL_TIMEOUT=180
MAX_RETRIES=3
POLL_INTERVAL=30
GPU_TYPE="NVIDIA GeForce RTX 4090"

# --- Parse args ---
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--gpu-type \"GPU NAME\"] <output_dir> [-- training_args...]"
    exit 1
fi
OUTPUT_DIR="$1"
shift

# Collect training args after "--"
TRAINING_ARGS=""
if [[ "${1:-}" == "--" ]]; then
    shift
    TRAINING_ARGS="$*"
fi

echo "Training remote run"
echo "  GPU type:   $GPU_TYPE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Train args: $TRAINING_ARGS"

# --- Setup ---
check_python_deps huggingface_hub
read_credentials

POD_ID=""
setup_pod_cleanup
mkdir -p "$OUTPUT_DIR"

# Record adapter repo timestamp before training so we can verify it was updated
BEFORE_TS=$(HF_TOKEN="$HF_TOKEN" REPO="$ADAPTER_REPO" python3 -c "
import os
from huggingface_hub import repo_info
try:
    info = repo_info(os.environ['REPO'], token=os.environ['HF_TOKEN'])
    print(info.last_modified.isoformat())
except Exception:
    print('NONE')
")
echo "Adapter repo last modified: $BEFORE_TS"

# --- Launch pod and poll ---
LAUNCH_CMD="\"$SCRIPT_DIR/launch_pod.sh\" train --gpu-type \"$GPU_TYPE\" $TRAINING_ARGS"
poll_pod "$LAUNCH_CMD" "$TIMEOUT" "$STALL_TIMEOUT" "$MAX_RETRIES" "$POLL_INTERVAL"

# --- Verify adapter repo was updated ---
echo ""
echo "=== Verify adapter updated on HF Hub ==="
AFTER_TS=$(HF_TOKEN="$HF_TOKEN" REPO="$ADAPTER_REPO" python3 -c "
import os
from huggingface_hub import repo_info
info = repo_info(os.environ['REPO'], token=os.environ['HF_TOKEN'])
print(info.last_modified.isoformat())
")
echo "  Before: $BEFORE_TS"
echo "  After:  $AFTER_TS"

if [[ "$BEFORE_TS" == "$AFTER_TS" && "$BEFORE_TS" != "NONE" ]]; then
    echo "  WARNING: Adapter repo timestamp unchanged — training may have failed."
    echo "  Check pod logs for errors."
    exit 1
fi
echo "  OK: Adapter repo was updated."

# --- Download training_metrics.json ---
echo ""
echo "=== Download training metrics ==="
if ! HF_TOKEN="$HF_TOKEN" REPO="$ADAPTER_REPO" OUTPUT="$OUTPUT_DIR/train_metrics.json" python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ['REPO'],
    filename='training_metrics.json',
    token=os.environ['HF_TOKEN'],
    force_download=True,
)
shutil.copy2(path, os.environ['OUTPUT'])
print(f'  Downloaded to {os.environ[\"OUTPUT\"]}')
"; then
    echo "ERROR: Failed to download training_metrics.json from $ADAPTER_REPO."
    echo "  The adapter may have been pushed without the metrics file."
    exit 1
fi

echo ""
echo "Done. Training metrics saved to $OUTPUT_DIR/train_metrics.json"
