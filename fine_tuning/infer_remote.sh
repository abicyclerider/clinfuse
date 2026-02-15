#!/usr/bin/env bash
# Automated remote inference round-trip:
#   upload Parquet → HF Hub → RunPod inference → HF Hub → download predictions
#
# Usage: ./infer_remote.sh [--gpu-type "GPU NAME"] <input_file> <output_file>
#        ./infer_remote.sh --local <input_file> <output_file>
#
# Features:
#   - --local: run inference locally on Apple Silicon (MPS) — no RunPod needed
#   - Retries up to 3 times if pod stalls on image pull (3 min stall timeout)
#   - Retries pod creation if GPU type is unavailable
#   - Terminates any created pod on exit/error/interrupt (trap on EXIT+signals)
#   - Validates predictions before declaring success

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_helpers.sh"

# Note: HF Hub repos use "grey" spelling (external resource, can't rename)
HF_INPUT_REPO="abicyclerider/grey-zone-pairs"
HF_OUTPUT_REPO="abicyclerider/grey-zone-predictions"
POLL_INTERVAL=30
TIMEOUT=1800  # 30 minutes
STALL_TIMEOUT=180  # 3 minutes — if container hasn't started, assume image pull stalled
MAX_RETRIES=3
GPU_TYPE="NVIDIA GeForce RTX 4090"
LOCAL_MODE=false

# --- Parse args ---
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        --local) LOCAL_MODE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [--local | --gpu-type \"GPU NAME\"] <input_file> <output_file>"
    exit 1
fi
INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# --- Local mode: run inference directly on this machine (MPS/CUDA) ---
if [[ "$LOCAL_MODE" == "true" ]]; then
    echo "Running inference locally..."
    echo "Input: $INPUT_FILE"
    mkdir -p "$(dirname "$OUTPUT_FILE")"

    if ! python3 "$SCRIPT_DIR/infer_classifier.py" \
        --input-file "$INPUT_FILE" \
        --output-file "$OUTPUT_FILE" \
        --no-quantize; then
        echo "ERROR: Local inference failed."
        exit 1
    fi

    if [[ ! -f "$OUTPUT_FILE" ]]; then
        echo "ERROR: Output file was not created: $OUTPUT_FILE"
        exit 1
    fi

    echo ""
    echo "Done. Predictions saved to $OUTPUT_FILE"
    exit 0
fi

# --- Remote mode: RunPod inference via HF Hub ---

check_python_deps datasets pandas huggingface_hub
read_credentials

INPUT_ROWS=$(INPUT_FILE="$INPUT_FILE" python3 -c "import os, pandas as pd; print(len(pd.read_parquet(os.environ['INPUT_FILE'])))")
echo "Input: $INPUT_FILE ($INPUT_ROWS rows)"
echo "GPU type: $GPU_TYPE"

# --- Cleanup: always terminate any pod we created ---
POD_ID=""
setup_pod_cleanup

# --- Step 1: Upload input to HF Hub ---
echo ""
echo "=== Step 1: Upload input to HF Hub ($HF_INPUT_REPO) ==="
if ! HF_TOKEN="$HF_TOKEN" INPUT_FILE="$INPUT_FILE" HF_REPO="$HF_INPUT_REPO" python3 -c "
import os
from datasets import Dataset
import pandas as pd
df = pd.read_parquet(os.environ['INPUT_FILE'])
Dataset.from_pandas(df).push_to_hub(os.environ['HF_REPO'])
print(f'  Uploaded {len(df)} rows')
"; then
    echo "ERROR: Failed to upload input to HF Hub."
    echo "  Check your HF_TOKEN and network connection."
    exit 1
fi

# --- Steps 2-3: Launch pod and poll (with retry on stall/failure) ---
LAUNCH_CMD="\"$SCRIPT_DIR/launch_pod.sh\" infer --gpu-type \"$GPU_TYPE\" --hf-input \"$HF_INPUT_REPO\" --hf-output \"$HF_OUTPUT_REPO\""
poll_pod "$LAUNCH_CMD" "$TIMEOUT" "$STALL_TIMEOUT" "$MAX_RETRIES" "$POLL_INTERVAL"

# --- Step 4: Download predictions from HF Hub ---
echo ""
echo "=== Step 4: Download predictions from HF Hub ($HF_OUTPUT_REPO) ==="
mkdir -p "$(dirname "$OUTPUT_FILE")"
if ! HF_TOKEN="$HF_TOKEN" OUTPUT_FILE="$OUTPUT_FILE" HF_REPO="$HF_OUTPUT_REPO" python3 -c "
import os
from datasets import load_dataset
ds = load_dataset(os.environ['HF_REPO'], split='train', download_mode='force_redownload')
df = ds.to_pandas()
df.to_parquet(os.environ['OUTPUT_FILE'], index=False)
print(f'  Downloaded {len(df)} rows')
"; then
    echo "ERROR: Failed to download predictions from HF Hub ($HF_OUTPUT_REPO)."
    echo "  The pod finished but predictions may not have been uploaded."
    echo "  Check pod logs for inference errors."
    exit 1
fi

if [[ ! -f "$OUTPUT_FILE" ]]; then
    echo "ERROR: Output file was not created: $OUTPUT_FILE"
    exit 1
fi

# --- Step 5: Validate output ---
echo ""
echo "=== Step 5: Validate predictions ==="
if ! OUTPUT_FILE="$OUTPUT_FILE" INPUT_ROWS="$INPUT_ROWS" python3 -c "
import os, sys
import pandas as pd
df = pd.read_parquet(os.environ['OUTPUT_FILE'])
expected = int(os.environ['INPUT_ROWS'])
errors = []
if 'prediction' not in df.columns:
    errors.append('Missing column: prediction')
if 'confidence' not in df.columns:
    errors.append('Missing column: confidence')
if len(df) != expected:
    errors.append(f'Row count mismatch: expected {expected}, got {len(df)}')
if errors:
    for e in errors:
        print(f'  FAIL: {e}')
    sys.exit(1)
match = (df['prediction'] == 1).sum()
nonmatch = (df['prediction'] == 0).sum()
print(f'  OK: {len(df)} rows with prediction + confidence')
print(f'  Predictions: {match} match, {nonmatch} non-match')
print(f'  Mean confidence: {df[\"confidence\"].mean():.4f}')
"; then
    echo "ERROR: Prediction validation failed. See errors above."
    echo "  The predictions file may be incomplete or malformed."
    exit 1
fi

echo ""
echo "Done. Predictions saved to $OUTPUT_FILE"
