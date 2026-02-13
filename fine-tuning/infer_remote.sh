#!/usr/bin/env bash
# Automated remote inference round-trip:
#   upload CSV → HF Hub → RunPod inference → HF Hub → download predictions
#
# Usage: ./infer_remote.sh [--gpu-type "GPU NAME"] <input_csv> <output_csv>
#        ./infer_remote.sh --local <input_csv> <output_csv>
#
# Features:
#   - --local: run inference locally on Apple Silicon (MPS) — no RunPod needed
#   - Retries up to 3 times if pod stalls on image pull (3 min stall timeout)
#   - Retries pod creation if GPU type is unavailable
#   - Terminates any created pod on exit/error/interrupt (trap on EXIT+signals)
#   - Validates predictions before declaring success

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    echo "Usage: $0 [--local | --gpu-type \"GPU NAME\"] <input_csv> <output_csv>"
    exit 1
fi
INPUT_CSV="$1"
OUTPUT_CSV="$2"

if [[ ! -f "$INPUT_CSV" ]]; then
    echo "ERROR: Input CSV not found: $INPUT_CSV"
    exit 1
fi

# --- Local mode: run inference directly on this machine (MPS/CUDA) ---
if [[ "$LOCAL_MODE" == "true" ]]; then
    echo "Running inference locally..."
    echo "Input: $INPUT_CSV"
    mkdir -p "$(dirname "$OUTPUT_CSV")"

    if ! python3 "$SCRIPT_DIR/inference_classifier.py" \
        --input-csv "$INPUT_CSV" \
        --output-csv "$OUTPUT_CSV" \
        --no-quantize; then
        echo "ERROR: Local inference failed."
        exit 1
    fi

    if [[ ! -f "$OUTPUT_CSV" ]]; then
        echo "ERROR: Output file was not created: $OUTPUT_CSV"
        exit 1
    fi

    echo ""
    echo "Done. Predictions saved to $OUTPUT_CSV"
    exit 0
fi

# --- Remote mode: RunPod inference via HF Hub ---

# --- Check Python dependencies ---
python3 -c "import datasets, pandas, huggingface_hub" 2>/dev/null || {
    echo "ERROR: Required Python packages: datasets, pandas, huggingface_hub"
    echo "Install: pip install datasets pandas huggingface_hub"
    exit 1
}

# --- Read credentials ---
ENV_FILE="$SCRIPT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found. Create it with HF_TOKEN=hf_..."
    exit 1
fi
HF_TOKEN=$(sed -n 's/^HF_TOKEN=//p' "$ENV_FILE" | tr -d '"' | tr -d "'")
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN not found in $ENV_FILE"
    exit 1
fi

RUNPOD_CONFIG="$HOME/.runpod/config.toml"
if [[ ! -f "$RUNPOD_CONFIG" ]]; then
    echo "ERROR: $RUNPOD_CONFIG not found. Run: runpodctl config --apiKey YOUR_KEY"
    exit 1
fi
RUNPOD_API_KEY=$(sed -n 's/^apikey = "\(.*\)"/\1/p' "$RUNPOD_CONFIG")
if [[ -z "$RUNPOD_API_KEY" ]]; then
    echo "ERROR: apikey not found in $RUNPOD_CONFIG"
    exit 1
fi

INPUT_ROWS=$(INPUT_CSV="$INPUT_CSV" python3 -c "import os, pandas as pd; print(len(pd.read_csv(os.environ['INPUT_CSV'])))")
echo "Input: $INPUT_CSV ($INPUT_ROWS rows)"
echo "GPU type: $GPU_TYPE"

# --- Cleanup: always terminate any pod we created ---
POD_ID=""
CLEANUP_DONE=false
cleanup() {
    # Guard against running cleanup twice (EXIT fires after signal traps)
    if [[ "$CLEANUP_DONE" == "true" ]]; then
        return
    fi
    CLEANUP_DONE=true

    if [[ -n "$POD_ID" ]]; then
        echo ""
        echo "=== Cleanup: Terminating pod $POD_ID ==="
        if ~/bin/runpodctl remove pod "$POD_ID" 2>/dev/null; then
            echo "  Pod $POD_ID terminated."
        else
            echo "  WARNING: Failed to terminate pod $POD_ID."
            echo "  Manually terminate: runpodctl remove pod $POD_ID"
        fi
        POD_ID=""
    fi
}
trap cleanup EXIT INT TERM HUP

# Helper: terminate current pod (used during retries, not final cleanup)
terminate_current_pod() {
    if [[ -n "$POD_ID" ]]; then
        echo "  Terminating pod $POD_ID..."
        ~/bin/runpodctl remove pod "$POD_ID" 2>/dev/null || true
        POD_ID=""
    fi
}

# Helper: query pod status, outputs "STATUS UPTIME" (uptime=-1 if container not started)
query_pod_status() {
    local pod_id="$1"
    local status_json
    status_json=$(curl -s --max-time 15 -X POST "https://api.runpod.io/graphql" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\":\"query { pod(input: {podId: \\\"$pod_id\\\"}) { id desiredStatus runtime { uptimeInSeconds } } }\"}") || {
        echo "QUERY_FAILED -1"
        return
    }
    echo "$status_json" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    pod = d.get('data', {}).get('pod')
    if pod is None:
        print('TERMINATED -1')
    else:
        status = pod.get('desiredStatus', 'UNKNOWN')
        rt = pod.get('runtime')
        uptime = rt.get('uptimeInSeconds', 0) if rt else -1
        print(f'{status} {uptime}')
except Exception:
    print('PARSE_ERROR -1')
" 2>/dev/null || echo "PARSE_ERROR -1"
}

# --- Step 1: Upload input CSV to HF Hub ---
echo ""
echo "=== Step 1: Upload input to HF Hub ($HF_INPUT_REPO) ==="
if ! HF_TOKEN="$HF_TOKEN" INPUT_CSV="$INPUT_CSV" HF_REPO="$HF_INPUT_REPO" python3 -c "
import os
from datasets import Dataset
import pandas as pd
df = pd.read_csv(os.environ['INPUT_CSV'])
Dataset.from_pandas(df).push_to_hub(os.environ['HF_REPO'])
print(f'  Uploaded {len(df)} rows')
"; then
    echo "ERROR: Failed to upload input CSV to HF Hub."
    echo "  Check your HF_TOKEN and network connection."
    exit 1
fi

# --- Steps 2-3: Launch pod and poll (with retry on stall/failure) ---
POD_SUCCEEDED=false
for ATTEMPT in $(seq 1 "$MAX_RETRIES"); do
    echo ""
    echo "=== Step 2: Launch RunPod pod (attempt $ATTEMPT/$MAX_RETRIES, GPU: $GPU_TYPE) ==="

    if ! POD_OUTPUT=$("$SCRIPT_DIR/run_on_runpod.sh" infer \
        --gpu-type "$GPU_TYPE" \
        --hf-input "$HF_INPUT_REPO" \
        --hf-output "$HF_OUTPUT_REPO" 2>&1); then
        echo "$POD_OUTPUT"
        echo ""
        echo "ERROR: Failed to create pod."
        # Check if the error mentions GPU availability
        if echo "$POD_OUTPUT" | grep -qi "no gpu\|no available\|insufficient\|out of stock"; then
            echo "  GPU type '$GPU_TYPE' appears to be unavailable on RunPod."
        else
            echo "  GPU type '$GPU_TYPE' may not be available, or there may be an API issue."
        fi
        echo "  Check: https://www.runpod.io/console/gpu-cloud"
        if [[ $ATTEMPT -lt $MAX_RETRIES ]]; then
            echo "  Retrying in 30s..."
            sleep 30
            continue
        fi
        echo "  All $MAX_RETRIES attempts failed."
        exit 1
    fi
    echo "$POD_OUTPUT"

    POD_ID=$(echo "$POD_OUTPUT" | sed -n 's/^Pod created: //p')
    if [[ -z "$POD_ID" ]]; then
        echo "ERROR: Failed to parse pod ID from run_on_runpod.sh output."
        echo "  Raw output was:"
        echo "$POD_OUTPUT"
        exit 1
    fi

    # --- Step 3: Poll pod status ---
    echo ""
    echo "=== Step 3: Polling pod status (stall: ${STALL_TIMEOUT}s, timeout: ${TIMEOUT}s) ==="
    ELAPSED=0
    STALLED=false
    POLL_ERRORS=0
    while true; do
        read -r POD_STATUS RUNTIME_UP <<< "$(query_pod_status "$POD_ID")"

        # Handle API query failures gracefully
        if [[ "$POD_STATUS" == "QUERY_FAILED" || "$POD_STATUS" == "PARSE_ERROR" ]]; then
            POLL_ERRORS=$((POLL_ERRORS + 1))
            echo "  [${ELAPSED}s] WARNING: API query failed (${POLL_ERRORS} consecutive)"
            if [[ $POLL_ERRORS -ge 5 ]]; then
                echo "  ERROR: Too many consecutive API failures. Aborting."
                exit 1
            fi
            sleep "$POLL_INTERVAL"
            ELAPSED=$((ELAPSED + POLL_INTERVAL))
            continue
        fi
        POLL_ERRORS=0  # reset on success

        if [[ "$RUNTIME_UP" == "-1" ]]; then
            echo "  [${ELAPSED}s] Status: $POD_STATUS (container not started)"
        else
            echo "  [${ELAPSED}s] Status: $POD_STATUS (uptime: ${RUNTIME_UP}s)"
        fi

        # Success: pod finished
        if [[ "$POD_STATUS" == "STOPPED" || "$POD_STATUS" == "EXITED" ]]; then
            echo "  Pod finished."
            POD_SUCCEEDED=true
            break
        fi

        # Pod disappeared
        if [[ "$POD_STATUS" == "TERMINATED" ]]; then
            echo "  WARNING: Pod was terminated unexpectedly."
            POD_ID=""
            if [[ $ATTEMPT -lt $MAX_RETRIES ]]; then
                echo "  Will retry..."
                STALLED=true
                break
            fi
            echo "  ERROR: Pod terminated on all attempts."
            exit 1
        fi

        # Stall detection: container hasn't started after STALL_TIMEOUT
        if [[ "$RUNTIME_UP" == "-1" && $ELAPSED -ge $STALL_TIMEOUT ]]; then
            echo "  Pod stalled — container not started after ${STALL_TIMEOUT}s (likely image pull issue)."
            terminate_current_pod
            STALLED=true
            break
        fi

        # Overall timeout
        if [[ $ELAPSED -ge $TIMEOUT ]]; then
            echo "  ERROR: Pod timed out after ${TIMEOUT}s."
            echo "  The container was running but inference didn't complete."
            echo "  Check logs: https://www.runpod.io/console/pods"
            exit 1
        fi

        sleep "$POLL_INTERVAL"
        ELAPSED=$((ELAPSED + POLL_INTERVAL))
    done

    if [[ "$STALLED" == "true" ]]; then
        if [[ $ATTEMPT -lt $MAX_RETRIES ]]; then
            echo "  Retrying with a new pod..."
            continue
        fi
        echo "ERROR: Pod stalled/failed on all $MAX_RETRIES attempts."
        exit 1
    fi

    # If we got here, pod succeeded
    break
done

if [[ "$POD_SUCCEEDED" != "true" ]]; then
    echo "ERROR: Inference did not complete successfully."
    exit 1
fi

# --- Step 4: Download predictions from HF Hub ---
echo ""
echo "=== Step 4: Download predictions from HF Hub ($HF_OUTPUT_REPO) ==="
mkdir -p "$(dirname "$OUTPUT_CSV")"
if ! HF_TOKEN="$HF_TOKEN" OUTPUT_CSV="$OUTPUT_CSV" HF_REPO="$HF_OUTPUT_REPO" python3 -c "
import os
from datasets import load_dataset
ds = load_dataset(os.environ['HF_REPO'], split='train', download_mode='force_redownload')
df = ds.to_pandas()
df.to_csv(os.environ['OUTPUT_CSV'], index=False)
print(f'  Downloaded {len(df)} rows')
"; then
    echo "ERROR: Failed to download predictions from HF Hub ($HF_OUTPUT_REPO)."
    echo "  The pod finished but predictions may not have been uploaded."
    echo "  Check pod logs for inference errors."
    exit 1
fi

if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "ERROR: Output file was not created: $OUTPUT_CSV"
    exit 1
fi

# --- Step 5: Validate output ---
echo ""
echo "=== Step 5: Validate predictions ==="
if ! OUTPUT_CSV="$OUTPUT_CSV" INPUT_ROWS="$INPUT_ROWS" python3 -c "
import os, sys
import pandas as pd
df = pd.read_csv(os.environ['OUTPUT_CSV'])
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
echo "Done. Predictions saved to $OUTPUT_CSV"
