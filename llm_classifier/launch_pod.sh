#!/usr/bin/env bash
# Launch an LLM classifier pipeline stage on RunPod GPU.
#
# Usage:
#   ./launch_pod.sh infer --hf-input abicyclerider/grey-zone-pairs --hf-output abicyclerider/grey-zone-predictions  # "grey" spelling in HF repo names
#   ./launch_pod.sh train --epochs 3 --batch-size 4
#   ./launch_pod.sh export --validate
#
# Reads HF_TOKEN from .env (same directory) and RUNPOD_API_KEY from ~/.runpod/config.toml.
# Creates a pod via the RunPod GraphQL API with the GHCR Docker image.
# The pod auto-stops after the script finishes (via RUNPOD_POD_ID + RUNPOD_API_KEY env vars).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Read the verified image tag from DVC's build_gpu_image stage output.
# This avoids race conditions where git HEAD points to a commit whose
# GHCR image hasn't been pushed yet (e.g. dvc.yaml-only changes).
IID_FILE="$(cd "$SCRIPT_DIR/.." && pwd)/.build/gpu-image.iid"
if [[ -f "$IID_FILE" ]]; then
    IMAGE="ghcr.io/abicyclerider/medgemma-pipeline:$(cat "$IID_FILE")"
else
    echo "ERROR: $IID_FILE not found. Run 'dvc repro build_gpu_image' first."
    exit 1
fi
DEFAULT_GPU="NVIDIA GeForce RTX 4090"
DEFAULT_DISK=30

# --- Parse stage name and forward remaining args ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <stage> [args...]"
    echo "Stages: infer, train, export"
    exit 1
fi

STAGE="$1"
shift

case "$STAGE" in
    infer)  SCRIPT="infer_classifier.py" ;;
    train)  SCRIPT="train_classifier.py" ;;
    export) SCRIPT="export_model.py" ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid stages: infer, train, export"
        exit 1
        ;;
esac

# Allow overrides via flags (must come after stage)
GPU_TYPE="$DEFAULT_GPU"
CONTAINER_DISK="$DEFAULT_DISK"
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        --container-disk) CONTAINER_DISK="$2"; shift 2 ;;
        *) break ;;  # Unknown flag — pass through to Python script
    esac
done

# Build the Python command with forwarded args
DOCKER_CMD="python $SCRIPT $*"

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

# --- Create pod via GraphQL API ---
echo "Creating RunPod pod..."
echo "  Image:   $IMAGE"
echo "  GPU:     $GPU_TYPE"
echo "  Disk:    ${CONTAINER_DISK}GB"
echo "  Command: $DOCKER_CMD"
echo ""

# Escape double quotes in the command for JSON embedding
DOCKER_CMD_ESCAPED=$(echo "$DOCKER_CMD" | sed 's/"/\\"/g')

RESPONSE=$(curl -s -X POST "https://api.runpod.io/graphql" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d "{\"query\":\"mutation { podFindAndDeployOnDemand(input: { name: \\\"medgemma-$STAGE\\\", imageName: \\\"$IMAGE\\\", gpuTypeId: \\\"$GPU_TYPE\\\", gpuCount: 1, containerDiskInGb: $CONTAINER_DISK, volumeInGb: 0, dockerArgs: \\\"$DOCKER_CMD_ESCAPED\\\", env: [ {key: \\\"HF_TOKEN\\\", value: \\\"$HF_TOKEN\\\"}, {key: \\\"RUNPOD_API_KEY\\\", value: \\\"$RUNPOD_API_KEY\\\"} ] }) { id imageName machine { podHostId } } }\"}")

# Check for errors
if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'data' in d and d['data']['podFindAndDeployOnDemand'] else 1)" 2>/dev/null; then
    POD_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['podFindAndDeployOnDemand']['id'])")
    echo "Pod created: $POD_ID"
    echo ""
    echo "Monitor status:"
    echo "  runpodctl get pod"
    echo ""
    echo "View logs (once running):"
    echo "  https://www.runpod.io/console/pods/$POD_ID/logs"
    echo ""
    echo "The pod will auto-stop when the script finishes."
    echo "To manually stop:  runpodctl stop pod $POD_ID"
    echo "To terminate:      runpodctl remove pod $POD_ID"
else
    echo "ERROR: Failed to create pod"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    exit 1
fi
