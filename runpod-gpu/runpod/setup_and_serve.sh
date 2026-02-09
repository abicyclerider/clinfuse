#!/bin/bash
# setup_and_serve.sh — Run this ON the RunPod Pod's web terminal.
# Downloads MedGemma 27B and starts a vLLM OpenAI-compatible server.
set -euo pipefail

MODEL="google/medgemma-27b-text-it"

echo "=== MedGemma 27B Pod Setup ==="

# 1. Verify HF_TOKEN is set (should be configured in the Pod template)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Set it in the Pod template env vars, or run: export HF_TOKEN=hf_..."
    exit 1
fi

# 2. Verify GPU is available
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is a GPU attached?"
    exit 1
fi
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 3. Download model (better progress output than letting vLLM download on first request)
echo "Downloading model: $MODEL"
huggingface-cli download "$MODEL" --quiet
echo "Download complete."
echo ""

# 4. Start vLLM server
echo "Starting vLLM server on 0.0.0.0:8000..."
echo "The server will be accessible at: https://<POD_ID>-8000.proxy.runpod.net/v1"
echo ""
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --dtype bfloat16
