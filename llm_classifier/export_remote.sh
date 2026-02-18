#!/usr/bin/env bash
# Automated remote export round-trip:
#   Launch RunPod pod → merge LoRA → push merged model to HF Hub → download info
#
# Usage: ./export_remote.sh [--gpu-type "GPU NAME"] <output_dir>
# Example: ./export_remote.sh output/training/export

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_helpers.sh"

OUTPUT_REPO="abicyclerider/medgemma-4b-entity-resolution-text-only"
TIMEOUT=1800  # 30 minutes
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
    echo "Usage: $0 [--gpu-type \"GPU NAME\"] <output_dir>"
    exit 1
fi
OUTPUT_DIR="$1"

echo "Export remote run"
echo "  GPU type:   $GPU_TYPE"
echo "  Output dir: $OUTPUT_DIR"

# --- Check promotion decision ---
TRAIN_DIR="$(dirname "$OUTPUT_DIR")/train"
PROMOTION_FILE="$TRAIN_DIR/promotion_decision.json"
if [[ -f "$PROMOTION_FILE" ]]; then
    PROMOTED=$(python3 -c "import json; print(json.load(open('$PROMOTION_FILE')).get('promoted', True))")
    if [[ "$PROMOTED" == "False" || "$PROMOTED" == "false" ]]; then
        echo "=== Model not promoted — skipping export ==="
        mkdir -p "$OUTPUT_DIR"
        echo '{"skipped": true, "reason": "model_not_promoted"}' > "$OUTPUT_DIR/export_info.json"
        exit 0
    fi
fi

# --- Setup ---
check_python_deps huggingface_hub
read_credentials

POD_ID=""
setup_pod_cleanup
mkdir -p "$OUTPUT_DIR"

# Record merged model repo timestamp before export
BEFORE_TS=$(HF_TOKEN="$HF_TOKEN" REPO="$OUTPUT_REPO" python3 -c "
import os
from huggingface_hub import repo_info
try:
    info = repo_info(os.environ['REPO'], token=os.environ['HF_TOKEN'])
    print(info.last_modified.isoformat())
except Exception:
    print('NONE')
")
echo "Merged model repo last modified: $BEFORE_TS"

# --- Launch pod and poll ---
LAUNCH_CMD="\"$SCRIPT_DIR/launch_pod.sh\" export --gpu-type \"$GPU_TYPE\" --container-disk 50"
poll_pod "$LAUNCH_CMD" "$TIMEOUT" "$STALL_TIMEOUT" "$MAX_RETRIES" "$POLL_INTERVAL"

# --- Verify merged model repo was updated ---
echo ""
echo "=== Verify merged model updated on HF Hub ==="
AFTER_TS=$(HF_TOKEN="$HF_TOKEN" REPO="$OUTPUT_REPO" python3 -c "
import os
from huggingface_hub import repo_info
info = repo_info(os.environ['REPO'], token=os.environ['HF_TOKEN'])
print(info.last_modified.isoformat())
")
echo "  Before: $BEFORE_TS"
echo "  After:  $AFTER_TS"

if [[ "$BEFORE_TS" == "$AFTER_TS" && "$BEFORE_TS" != "NONE" ]]; then
    echo "  WARNING: Merged model repo timestamp unchanged — export may have failed."
    echo "  Check pod logs for errors."
    exit 1
fi
echo "  OK: Merged model repo was updated."

# --- Download export_info.json ---
echo ""
echo "=== Download export info ==="
if ! HF_TOKEN="$HF_TOKEN" REPO="$OUTPUT_REPO" OUTPUT="$OUTPUT_DIR/export_info.json" python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ['REPO'],
    filename='export_info.json',
    token=os.environ['HF_TOKEN'],
    force_download=True,
)
shutil.copy2(path, os.environ['OUTPUT'])
print(f'  Downloaded to {os.environ[\"OUTPUT\"]}')
"; then
    echo "ERROR: Failed to download export_info.json from $OUTPUT_REPO."
    echo "  The model may have been pushed without the info file."
    exit 1
fi

echo ""
echo "Done. Export info saved to $OUTPUT_DIR/export_info.json"
